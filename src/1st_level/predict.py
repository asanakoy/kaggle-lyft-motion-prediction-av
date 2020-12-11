import argparse
import os
import pprint
import numpy as np
import torch
import torch.cuda.amp
import torch.distributed
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from common import *
from models import build_model
from utils import DotDict


def prepare_submission(experiment_name, epoch, stage):
    model_str = experiment_name
    cfg = load_config_data(experiment_name)
    pprint.pprint(cfg)

    checkpoints_dir = f"./checkpoints/{model_str}"
    print("\n", experiment_name, "\n")

    model_info = DotDict(cfg["model_params"])
    model = build_model(model_info, cfg)
    model = model.cuda()
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    torch.set_grad_enabled(False)

    eval_dataset = dataset.LyftDatasetPrerendered(dset_name=stage, cfg_data=cfg)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--epoch", type=int, default=-1)

    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--nb_batches", type=int, default=1)

    args = parser.parse_args()
    action = args.action

    if action == "prepare_submission":
        prepare_submission(
            experiment_name=normalize_experiment_name(args.experiment),
            epoch=args.epoch,
            stage=dataset.LyftDataset.STAGE_TEST,
        )

    if action == "prepare_submission_val":
        prepare_submission(
            experiment_name=normalize_experiment_name(args.experiment),
            epoch=args.epoch,
            stage=dataset.LyftDataset.STAGE_VALIDATION,
        )
