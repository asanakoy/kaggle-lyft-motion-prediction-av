import argparse
import os
import pprint

import numpy as np
import torch
import torch.cuda.amp
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
from adamp import AdamP
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset
import utils
from common import *
from models import build_model
from utils import DotDict


def train(experiment_name, distributed=False, continue_epoch=-1):
    model_str = experiment_name

    cfg = load_config_data(experiment_name)
    pprint.pprint(cfg)

    model_type = cfg["model_params"]["model_type"]
    train_params = DotDict(cfg["train_params"])

    checkpoints_dir = f"./checkpoints/{model_str}"
    tensorboard_dir = f"./tensorboard/{model_type}/{model_str}"
    oof_dir = f"./oof/{model_str}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print("\n", experiment_name, "\n")

    logger = SummaryWriter(log_dir=tensorboard_dir)

    scaler = torch.cuda.amp.GradScaler()

    with utils.timeit_context("load train"):
        dataset_train = dataset.LyftDatasetPrerendered(dset_name=dataset.LyftDataset.DSET_TRAIN_XXL, cfg_data=cfg)

    with utils.timeit_context("load validation"):
        dataset_valid = dataset.LyftDatasetPrerendered(dset_name=dataset.LyftDataset.DSET_VALIDATION, cfg_data=cfg)

    batch_size = dataset_train.dset_cfg["batch_size"]

    data_loaders = {
        "train": DataLoader(dataset_train, num_workers=16, shuffle=True, batch_size=batch_size),
        "val": DataLoader(
            dataset_valid,
            shuffle=False,
            num_workers=16,
            batch_size=dataset_valid.dset_cfg["batch_size"],
        ),
    }
    model_info = DotDict(cfg["model_params"])
    model = build_model(model_info, cfg)
    model = model.cuda()

    model.train()

    initial_lr = float(train_params.initial_lr)
    if train_params.optimizer == "adamp":
        optimizer = AdamP(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "sgd":
        if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_EMB:
            optimizer = optim.SGD(
                [
                    {
                        "params": [
                            v
                            for n, v in model.named_parameters()
                            if not n.startswith("emb.") and not n.startswith("backbone.")
                        ],
                        "lr": initial_lr * 2,
                    },
                    {"params": model.backbone.parameters(), "lr": initial_lr},
                    {"params": model.emb.parameters(), "lr": initial_lr * 20},
                ],
                lr=initial_lr,
                momentum=0.9,
                nesterov=True,
            )
        else:
            optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError("Invalid optimiser" + train_params.optimizer)

    if continue_epoch > 0:
        checkpoint = torch.load(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    nb_epochs = train_params.nb_epochs
    if train_params.scheduler == "steps":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_params.optimiser_milestones,
            gamma=0.2,
            last_epoch=continue_epoch,
        )
    elif train_params.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=nb_epochs,
            eta_min=initial_lr / 1000,
            last_epoch=continue_epoch,
        )
    elif train_params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = utils.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_params.scheduler_period,
            T_mult=train_params.get('scheduler_t_mult', 1),
            eta_min=initial_lr / 1000.0,
            last_epoch=-1
        )
        for i in range(continue_epoch + 1):
            scheduler.step()
    else:
        raise RuntimeError("Invalid scheduler name")

    grad_clip_value = train_params.get("grad_clip", 2.0)
    print("grad clip:", grad_clip_value)

    print(f"Num training agents: {len(dataset_train)} validation agents: {len(dataset_valid)}")

    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        for phase in ["train", "val"]:
            model.train(phase == "train")
            epoch_loss_segmentation = []
            epoch_loss_regression = []
            epoch_loss_regression_aux = []
            data_loader = data_loaders[phase]

            optimizer.zero_grad()

            if phase == "train":
                nb_steps_per_epoch = train_params.epoch_size // batch_size
                data_iter = tqdm(
                    utils.LoopIterable(data_loader, max_iters=nb_steps_per_epoch),
                    total=nb_steps_per_epoch,
                    ncols=250,
                )
            else:
                if epoch_num % 2 > 0:  # skip each 4th validation for speed
                    continue

                data_iter = tqdm(data_loader, ncols=250)

            for data in data_iter:
                with torch.set_grad_enabled(phase == "train"):
                    # torch.set_anomaly_enabled(True)
                    inputs = data["image"].float().cuda()
                    # agent_state = data["agent_state"].float().cuda()
                    agent_state = None
                    target_availabilities = data["target_availabilities"].cuda()

                    targets = data["target_positions"].cuda()

                    pos_scale = 1.0

                    optimizer.zero_grad()

                    loss_segmentation = 0
                    loss_regression = 0
                    loss_regression_aux = 0

                    if model_type == MODEL_TYPE_ATTENTION:
                        all_agents_state = data["all_agents_state"].float().cuda()
                        image_blocks_positions_agent = data["image_blocks_positions_agent"].cuda()

                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, image_blocks_positions_agent, all_agents_state)

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                                gt=targets.float() * pos_scale,
                                pred=pred.float() * pos_scale,
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_WITH_OTHER_AGENTS_INPUTS:
                        all_agents_state = data["all_agents_state"].float().cuda()

                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, all_agents_state)

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                                gt=targets.float(),
                                pred=pred.float(),
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE:
                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, agent_state)

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch_from_log_sm(
                                gt=targets.float() * pos_scale,
                                pred=pred.float() * pos_scale,
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_AUX_OUT:
                        with torch.cuda.amp.autocast():
                            pred, confidences, pred_aux, confidences_aux = model(
                                inputs,
                                agent_state,
                                data["image_4x"].float().cuda()
                            )

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch_from_log_sm(
                                gt=targets.float(),
                                pred=pred.float(),
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                            loss_regression_aux = utils.pytorch_neg_multi_log_likelihood_batch_from_log_sm(
                                gt=targets.float(),
                                pred=pred_aux.float(),
                                confidences=confidences_aux.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_I4X:
                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, agent_state, data["image_4x"].float().cuda())

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                                gt=targets.float() * pos_scale,
                                pred=pred.float() * pos_scale,
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_WITH_MASKS:
                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, agent_state, data["other_agents_masks"].float().cuda())

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                                gt=targets.float() * pos_scale,
                                pred=pred.float() * pos_scale,
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_EMB:
                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, agent_state, data["corners"].float().cuda())

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                                gt=targets.float() * pos_scale,
                                pred=pred.float() * pos_scale,
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )
                    elif model_type == MODEL_TYPE_SEGMENTATION:
                        target_mask = data["output_mask"].cuda()
                        l2_cls, l1_cls = model(inputs, agent_state)
                        loss_segmentation = (
                                torch.nn.functional.binary_cross_entropy_with_logits(l2_cls, target_mask) * 1000
                                + torch.nn.functional.binary_cross_entropy_with_logits(l1_cls, target_mask) * 100
                        )
                    elif model_type == MODEL_TYPE_SEGMENTATION_AND_REGRESSION:
                        target_mask = data["output_mask"].cuda()
                        segmentation, pred, confidences = model(inputs, agent_state)
                        loss_segmentation = (
                                torch.nn.functional.binary_cross_entropy_with_logits(segmentation, target_mask) * 1000
                        )

                        loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                            gt=targets.float() * pos_scale,
                            pred=pred.float() * pos_scale,
                            confidences=confidences.float(),
                            avails=target_availabilities.float(),
                        )

                    loss = loss_segmentation + loss_regression + loss_regression_aux

                    if phase == "train":
                        scaler.scale(loss).backward()

                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer)
                        scaler.update()

                    if phase == "val":
                        # save predictions visualisation
                        pass

                    epoch_loss_segmentation.append(float(loss_segmentation))
                    epoch_loss_regression.append(float(loss_regression))
                    epoch_loss_regression_aux.append(float(loss_regression_aux))
                    loss_segmentation = None
                    loss_regression = None
                    loss_regression_aux = None
                    del loss

                    data_iter.set_description(
                        f"{epoch_num} {phase[0]}"
                        f" Loss r {np.mean(epoch_loss_regression):1.4f} "
                        f" r aux {np.mean(epoch_loss_regression_aux):1.4f} "
                        f"s {np.mean(epoch_loss_segmentation):1.4f}"
                    )

            logger.add_scalar(f"loss_{phase}", np.mean(epoch_loss_regression), epoch_num)
            if epoch_loss_segmentation[-1] > 0:
                logger.add_scalar(f"loss_segmentation_{phase}", np.mean(epoch_loss_segmentation), epoch_num)

            if epoch_loss_regression_aux[-1] > 0:
                logger.add_scalar(f"loss_regression_aux_{phase}", np.mean(epoch_loss_regression_aux), epoch_num)

            if phase == "train":
                logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
            logger.flush()

            if phase == "train":
                scheduler.step()
                if (epoch_num % train_params.save_period == 0) or (epoch_num == nb_epochs):
                    torch.save(
                        {
                            "epoch": epoch_num,
                            "model_state_dict": model.module.state_dict() if distributed else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        f"{checkpoints_dir}/{epoch_num:03}.pt",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--epoch", type=int, default=-1)

    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--nb_batches", type=int, default=1)

    args = parser.parse_args()
    action = args.action

    if action == "train":
        try:
            train(
                experiment_name=normalize_experiment_name(args.experiment),
                continue_epoch=args.epoch,
            )
        except KeyboardInterrupt:
            pass  # avoid printing large backtrace on keyboard interrupt
