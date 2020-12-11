import argparse
import os
import pprint

import numpy as np
import torch
import torch.cuda.amp
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from l5kit.evaluation import write_pred_csv, compute_metrics_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.geometry import transform_points
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset
import utils
from common import *
from models import build_model
from utils import DotDict


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '42355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train(rank, experiment_name, world_size, continue_epoch, dist_url):
    print(f"Running rank {rank}/{world_size} dist url: {dist_url}.")
    # setup(rank, world_size)

    dist.init_process_group(backend="nccl", init_method=dist_url,
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    distributed = True
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

    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=world_size)

    data_loaders = {
        "train": DataLoader(
            dataset_train,
            num_workers=16,
            shuffle=True,
            # sampler=train_sampler,
            batch_size=batch_size // world_size
        ),
        "val": DataLoader(
            dataset_valid,
            shuffle=False,
            num_workers=16,
            batch_size=dataset_valid.dset_cfg["batch_size"] // world_size,
        ),
    }
    model_info = DotDict(cfg["model_params"])
    model_orig = build_model(model_info, cfg).cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model_orig)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    initial_lr = float(train_params.initial_lr)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)

    if continue_epoch > 0:
        # if rank == 0:
        fn = f"{checkpoints_dir}/{continue_epoch:03}.pt"
        print(f'loading {fn}...')
        # dist.barrier()

        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(fn, map_location=map_location)

        # if distributed:
        #     model.module.load_state_dict(checkpoint["model_state_dict"])

        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        dist.barrier()
        print(f'loaded {fn}')

    nb_epochs = train_params.nb_epochs
    scheduler = utils.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_params.scheduler_period,
        T_mult=train_params.get('scheduler_t_mult', 1),
        eta_min=initial_lr / 1000.0,
        last_epoch=-1
    )
    for i in range(continue_epoch + 1):
        scheduler.step()

    grad_clip_value = train_params.get("grad_clip", 2.0)
    print("grad clip:", grad_clip_value)

    print(f"Num training agents: {len(dataset_train)} validation agents: {len(dataset_valid)}")

    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        for phase in ["train", "val"]:
            model.train(phase == "train")
            epoch_loss_regression = []
            data_loader = data_loaders[phase]

            optimizer.zero_grad()

            if phase == "train":
                nb_steps_per_epoch = train_params.epoch_size // batch_size
                data_iter = tqdm(
                    utils.LoopIterable(data_loader, max_iters=nb_steps_per_epoch),
                    total=nb_steps_per_epoch,
                    ncols=250,
                    # disable=rank > 0
                )
            else:
                if epoch_num % 2 == 1:  # skip each second validation for speed
                    continue

                data_iter = tqdm(data_loader, ncols=250)

            for data in data_iter:
                with torch.set_grad_enabled(phase == "train"):
                    inputs = data["image"].float().cuda(rank, non_blocking=True)
                    target_availabilities = data["target_availabilities"].cuda(rank, non_blocking=True)
                    targets = data["target_positions"].cuda(rank, non_blocking=True)

                    optimizer.zero_grad()
                    loss_regression = 0
                    agent_state = None

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE:
                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs)

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch_from_log_sm(
                                gt=targets.float(),
                                pred=pred.float(),
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    if model_type == MODEL_TYPE_REGRESSION_MULTI_MODE_I4X:
                        with torch.cuda.amp.autocast():
                            pred, confidences = model(inputs, agent_state, data["image_4x"].float().cuda())

                            loss_regression = utils.pytorch_neg_multi_log_likelihood_batch(
                                gt=targets.float(),
                                pred=pred.float(),
                                confidences=confidences.float(),
                                avails=target_availabilities.float(),
                            )

                    loss = loss_regression

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

                    epoch_loss_regression.append(float(loss_regression))
                    loss_regression = None
                    del loss

                    data_iter.set_description(
                        f"{epoch_num} {phase[0]}"
                        f" Loss r {np.mean(epoch_loss_regression):1.4f} "
                    )

            if rank == 0:
                logger.add_scalar(f"loss_{phase}", np.mean(epoch_loss_regression), epoch_num)

                if phase == "train":
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
                logger.flush()

            if phase == "train":
                scheduler.step()

                if rank == 0:
                    torch.save(
                        {
                            "epoch": epoch_num,
                            # "model_state_dict": model.state_dict(),
                            "model_state_dict": model.module.state_dict() if distributed else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        f"{checkpoints_dir}/{epoch_num:03}.pt",
                    )


def prepare_submission(rank, experiment_name, epoch, stage, dist_url, world_size):
    print(f"Running rank {rank}/{world_size} dist url: {dist_url}.")
    # setup(rank, world_size)

    dist.init_process_group(backend="nccl", init_method=dist_url,
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    distributed = True

    model_str = experiment_name
    cfg = load_config_data(experiment_name)
    pprint.pprint(cfg)

    checkpoints_dir = f"./checkpoints/{model_str}"
    print("\n", experiment_name, "\n")

    model_info = DotDict(cfg["model_params"])
    model_orig = build_model(model_info, cfg).cuda(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model_orig)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    model.eval()

    # if rank == 0:
    fn = f"{checkpoints_dir}/{epoch:03}.pt"
    print(f'loading {fn}...')

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(fn, map_location=map_location)

    model.module.load_state_dict(checkpoint["model_state_dict"])
    print(f'loaded {fn}')

    model.eval()
    torch.set_grad_enabled(False)

    eval_dataset = dataset.LyftDatasetPrerendered(stage=stage, cfg_data=cfg)
    # eval_dataset[0]

    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=32, num_workers=16)

    # print(eval_dataset.agent_dataset)

    def run_prediction(predictor, data_loader):
        predictor.eval()

        pred_coords_list = []
        confidences_list = []
        timestamps_list = []
        track_id_list = []

        with torch.no_grad():
            for data in tqdm(data_loader):
                image = data["image"].cuda()
                # agent_state = data["agent_state"].float().cuda()
                agent_state = None

                pred, confidences = predictor(image, agent_state)
                confidences = torch.exp(confidences)

                pred_world = []
                pred = pred.cpu().numpy().copy()
                if model_info.target_space == "image":
                    if model_info.target_space == "image":
                        world_from_agents = data["world_from_agent"].numpy()
                        centroids = data["centroid"].numpy()
                        for idx in range(pred.shape[0]):
                            pred[idx] = (
                                    transform_points(
                                        pred[idx].copy().reshape(-1, 2),
                                        world_from_agents[idx],
                                    )
                                    - centroids[idx]
                            ).reshape(-1, 50, 2)

                for img_idx in range(pred.shape[0]):
                    pred_world.append(pred[img_idx])

                pred_coords_list.append(np.array(pred_world))
                confidences_list.append(confidences.cpu().numpy().copy())
                timestamps_list.append(data["timestamp"].numpy().copy())
                track_id_list.append(data["track_id"].numpy().copy())

        timestamps = np.concatenate(timestamps_list)
        track_ids = np.concatenate(track_id_list)
        coords = np.concatenate(pred_coords_list)
        confs = np.concatenate(confidences_list)
        return timestamps, track_ids, coords, confs

    timestamps, track_ids, coords, confs = run_prediction(model, eval_dataloader)
    os.makedirs("submissions", exist_ok=True)
    pred_path = f"submissions/sub_{experiment_name}_{epoch}_{stage}.csv"

    print(f"Coords: {coords.shape} conf: {confs.shape}")
    np.savez_compressed(f"submissions/sub_{experiment_name}_{epoch}_{stage}.npz",
                        timestamps=timestamps,
                        track_ids=track_ids,
                        coords=coords,
                        confs=confs)

    write_pred_csv(
        pred_path,
        timestamps=timestamps,
        track_ids=track_ids,
        coords=coords,
        confs=confs,
    )
    print(f"Saved to {pred_path}")
    # metrics = compute_metrics_csv('../data/gt_val_100_chopped.csv', pred_path, [neg_multi_log_likelihood])
    # for metric_name, metric_mean in metrics.items():
    #     print(pred_path, metric_name, metric_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--epoch", type=int, default=-1)

    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--nb_batches", type=int, default=1)
    parser.add_argument("--rank", default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                        help='url used to set up distributed training')

    args = parser.parse_args()
    action = args.action

    if action == "train":
        # ngpus_per_node = torch.cuda.device_count()
        train(
            experiment_name=normalize_experiment_name(args.experiment),
            continue_epoch=args.epoch,
            rank=args.rank,
            world_size=args.world_size,
            dist_url=args.dist_url
        )

    if action == "prepare_submission":
        prepare_submission(
            rank=args.rank,
            world_size=args.world_size,
            dist_url=args.dist_url,
            experiment_name=normalize_experiment_name(args.experiment),
            epoch=args.epoch,
            stage=dataset.LyftDataset.STAGE_TEST,
        )

    if action == "prepare_submission_val":
        prepare_submission(
            rank=args.rank,
            world_size=args.world_size,
            dist_url=args.dist_url,
            experiment_name=normalize_experiment_name(args.experiment),
            epoch=args.epoch,
            stage=dataset.LyftDataset.STAGE_VALIDATION,
        )
