import glob
import logging
import math
import os
from collections import defaultdict
from enum import Enum
from os.path import join
from pathlib import Path

import cv2
import l5kit.data
import numpy as np
import torch
import zarr
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.evaluation import (
    read_gt_csv,
)
from l5kit.geometry import transform_point, transform_points
from l5kit.rasterization import build_rasterizer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

import config
import utils
from fast_rasteriser import build_custom_rasterizer

logger = logging.getLogger("dataset")
os.environ["L5KIT_DATA_FOLDER"] = config.L5KIT_DATA_FOLDER


def fix_agent_state(agent_data, agent_state):
    if agent_state is not None:
        if math.cos(agent_data["yaw"] - agent_state["yaw"]) < -0.5 and agent_state["velocity"] < 0:
            agent_state["yaw"] += math.pi
            agent_state["velocity"] *= -1
            agent_state["accel"] *= -1

    return agent_data, agent_state


class LyftDataset(torch.utils.data.Dataset):
    DSET_TRAIN = "train"
    DSET_TRAIN_XXL = "train_XXL"
    DSET_VALIDATION = "val"
    DSET_VALIDATION_CHOPPED = "val_chopped"
    DSET_TEST = "test"

    # for backward compatibility
    STAGE_TRAIN = "train"
    STAGE_VALIDATION = "val"
    STAGE_VALIDATION_CHOPPED = "val_chopped"
    STAGE_TEST = "test"

    name_2_dataloader_key = {
        DSET_TRAIN: "train_data_loader",
        DSET_TRAIN_XXL: "train_data_loader",
        DSET_VALIDATION: "val_data_loader",
        DSET_VALIDATION_CHOPPED: "val_data_loader",
        DSET_TEST: "test_data_loader",
    }

    def __init__(
            self,
            dset_name=None,
            cfg_path="./agent_motion_config.yaml",
            cfg_data=None,
            stage=None,
    ):
        print(f"Initializing LyftDataset {dset_name}...")
        if stage is not None:
            print('DDEPRECATION WARNING! LyftDataset:: argument "stage=" is deprecated, use "dset_name=" instead')
            if dset_name is None:
                dset_name = stage
            else:
                raise ValueError('LyftDataset::Please use only "dset_name" argument')
        assert dset_name is not None
        self.dm = LocalDataManager(None)
        self.dset_name = dset_name
        if cfg_data is None:
            self.cfg = utils.DotDict(load_config_data(cfg_path))
        else:
            self.cfg = utils.DotDict(cfg_data)

        self.dset_cfg = self.cfg[LyftDataset.name_2_dataloader_key[dset_name]].copy()

        if self.cfg["raster_params"]["map_type"] == "py_satellite":
            print("WARNING! USING SLOW RASTERIZER!!! py_satellite")
            self.rasterizer = build_rasterizer(self.cfg, self.dm)
        self.rasterizer = build_custom_rasterizer(self.cfg, self.dm)

        if dset_name == LyftDataset.DSET_VALIDATION_CHOPPED:
            eval_base_path = Path("/opt/data3/lyft_motion_prediction/prediction_dataset/scenes/validate_chopped_100")
            eval_zarr_path = str(Path(eval_base_path) / Path(self.dm.require(self.dset_cfg["key"])).name)
            eval_mask_path = str(Path(eval_base_path) / "mask.npz")
            self.eval_gt_path = str(Path(eval_base_path) / "gt.csv")
            self.zarr_dataset = ChunkedDataset(eval_zarr_path).open(cached=False)
            self.agent_dataset = AgentDataset(
                self.cfg,
                self.zarr_dataset,
                self.rasterizer,
                agents_mask=np.load(eval_mask_path)["arr_0"],
            )

            self.val_chopped_gt = defaultdict(dict)
            for el in read_gt_csv(self.eval_gt_path):
                self.val_chopped_gt[el["track_id"] + el["timestamp"]] = el
        elif dset_name == LyftDataset.DSET_TEST:
            self.zarr_dataset = ChunkedDataset(self.dm.require(self.dset_cfg["key"])).open(cached=False)
            test_mask = np.load(f"{config.L5KIT_DATA_FOLDER}/scenes/mask.npz")["arr_0"]
            self.agent_dataset = AgentDataset(self.cfg, self.zarr_dataset, self.rasterizer, agents_mask=test_mask)
        else:
            zarr_path = self.dm.require(self.dset_cfg["key"])
            print(f"Opening Chunked Dataset {zarr_path}...")
            self.zarr_dataset = ChunkedDataset(zarr_path).open(cached=False)
            print("Creating Agent Dataset...")
            self.agent_dataset = AgentDataset(
                self.cfg,
                self.zarr_dataset,
                self.rasterizer,
                min_frame_history=0,
                min_frame_future=10,
            )
            print("Creating Agent Dataset... [OK]")

        if dset_name == LyftDataset.DSET_VALIDATION:
            mask_frame100 = np.zeros(shape=self.agent_dataset.agents_mask.shape, dtype=np.bool)
            for scene in self.agent_dataset.dataset.scenes:
                frame_interval = scene["frame_index_interval"]
                agent_index_interval = self.agent_dataset.dataset.frames[frame_interval[0] + 99]["agent_index_interval"]
                mask_frame100[agent_index_interval[0]: agent_index_interval[1]] = True

            prev_agents_num = np.sum(self.agent_dataset.agents_mask)
            self.agent_dataset.agents_mask = self.agent_dataset.agents_mask * mask_frame100
            print(f"nb agent: orig {prev_agents_num} filtered {np.sum(self.agent_dataset.agents_mask)}")
            # store the valid agents indexes
            self.agent_dataset.agents_indices = np.nonzero(self.agent_dataset.agents_mask)[0]

        self.w, self.h = self.cfg["raster_params"]["raster_size"]

        self.add_agent_state = self.cfg["model_params"]["add_agent_state"]
        self.agent_state = None

    def __len__(self):
        return len(self.agent_dataset)

    def __getitem__(self, item_idx):
        data = self.agent_dataset[item_idx]
        return data


def pos_ahead(agent_data):
    time_ahead = 2.5
    distance_ahead = 5.0

    xy = agent_data["centroid"]
    vel = agent_data["velocity"]
    return xy + vel * time_ahead


class TLColor(Enum):
    unknown = -1
    green = 0
    yellow = 1
    red = 2

    green_left = 3
    green_right = 4
    yellow_left = 5
    yellow_right = 6
    red_left = 7
    red_right = 8


class LyftDatasetPrerendered(torch.utils.data.Dataset):
    def __init__(
            self,
            dset_name=None,
            cfg_path="./agent_motion_config.yaml",
            cfg_data=None,
            stage=None,
    ):
        if stage is not None:
            print('LyftDatasetPrerendered:: argument "stage=" is deprecated, use "dset_name=" instead')
            if dset_name is None:
                dset_name = stage
            else:
                raise ValueError('LyftDatasetPrerendered::Please use only "dset_name" argument')
        assert dset_name is not None
        logger.info(f"Initializing prerendered {dset_name} dataset...")
        self.dm = LocalDataManager(None)
        self.dset_name = dset_name
        if cfg_data is None:
            self.cfg = load_config_data(cfg_path)
        else:
            self.cfg = cfg_data

        # only used for rgb visualisation
        self.rasterizer = build_custom_rasterizer(self.cfg, self.dm)

        self.dset_cfg = self.cfg[LyftDataset.name_2_dataloader_key[dset_name]].copy()

        root_dir = self.dset_cfg.get("root_dir", None)
        if root_dir is None:
            data_dir_name = {
                LyftDataset.DSET_TRAIN: "train_uncompressed",
                LyftDataset.DSET_TRAIN_XXL: "train_XXL",
                LyftDataset.DSET_VALIDATION: "validate_uncompressed",
                LyftDataset.DSET_TEST: "test",
            }[dset_name]
            self.root_dir_name = join(
                config.L5KIT_DATA_FOLDER, self.cfg["raster_params"]["pre_render_cache_dir"], data_dir_name
            )
        else:
            self.root_dir_name = join(config.L5KIT_DATA_FOLDER, root_dir)
        print("load pre-rendered raster from", self.root_dir_name)

        self.segmentation_output = self.cfg["raster_params"].get("segmentation_output", None)
        self.segmentation_results_dir = self.cfg["raster_params"].get("segmentation_results_dir", None)
        self.add_own_agent_mask = self.cfg["raster_params"].get("add_own_agent_mask", False)
        print(f"Segmentation model res: {self.segmentation_results_dir} {self.add_own_agent_mask}")

        all_files_fn = join(self.root_dir_name, self.dset_cfg.get("filepaths_cache", "all_files") + ".npy")
        try:
            logger.info(f"Loading cached filenames from {all_files_fn}")
            self.all_files = np.load(all_files_fn, allow_pickle=True)
        except FileNotFoundError:
            logger.info(f"Generating and caching filenames in {all_files_fn}")
            self.all_files = list(sorted(glob.glob(f"{self.root_dir_name}/**/*.npz", recursive=True)))
            print(f"Generated all npz paths and saved to {all_files_fn}")
            np.save(all_files_fn, self.all_files)
        print(f"Found {len(self.all_files)} agents")
        self.add_agent_state = self.cfg["model_params"]["add_agent_state"]
        self.add_agent_state_history = self.cfg["model_params"].get("add_agent_state_history", False)
        self.agent_state_history_steps = self.cfg["model_params"].get("agent_state_history_steps", 20)
        self.max_agent_in_state_history = self.cfg["model_params"].get("max_agent_in_state_history", 16)
        self.w, self.h = self.cfg["raster_params"]["raster_size"]

        self.tf_face_colors = {}

        zarr_path = self.dm.require(self.dset_cfg["key"])
        print(f"Opening Chunked Dataset {zarr_path}...")
        # print("Creating Agent Dataset...")
        # self.agent_dataset = AgentDataset(
        #     self.cfg,
        #     self.zarr_dataset,
        #     self.rasterizer,
        #     min_frame_history=0,
        #     min_frame_future=10,
        # )

        if self.add_agent_state_history:
            self.zarr_dataset = ChunkedDataset(zarr_path).open()
            self.all_scenes = self.zarr_dataset.scenes[:].copy()
            self.all_frames_agent_interval = self.zarr_dataset.frames['agent_index_interval'].copy()
        print("Creating Agent Dataset... [OK]")

    def __len__(self):
        return len(self.all_files)

    def tl_element_color(self, element):
        if not element.element.HasField("traffic_control_element"):
            return TLColor.unknown

        traffic_el = element.element.traffic_control_element

        if traffic_el.HasField(f"signal_red_face"):
            return TLColor.red

        if traffic_el.HasField(f"signal_left_arrow_red_face"):
            return TLColor.red_left
        if traffic_el.HasField(f"signal_upper_left_arrow_red_face"):
            return TLColor.red_left

        if traffic_el.HasField(f"signal_right_arrow_red_face"):
            return TLColor.red_right
        if traffic_el.HasField(f"signal_upper_right_arrow_red_face"):
            return TLColor.red_right

        if traffic_el.HasField(f"signal_yellow_face"):
            return TLColor.yellow

        if traffic_el.HasField(f"signal_left_arrow_yellow_face"):
            return TLColor.yellow_left
        if traffic_el.HasField(f"signal_upper_left_arrow_yellow_face"):
            return TLColor.yellow_left

        if traffic_el.HasField(f"signal_right_arrow_yellow_face"):
            return TLColor.yellow_right
        if traffic_el.HasField(f"signal_upper_right_arrow_yellow_face"):
            return TLColor.yellow_right

        if traffic_el.HasField(f"signal_green_face"):
            return TLColor.green

        if traffic_el.HasField(f"signal_left_arrow_green_face"):
            return TLColor.green_left
        if traffic_el.HasField(f"signal_upper_left_arrow_green_face"):
            return TLColor.green_left

        if traffic_el.HasField(f"signal_right_arrow_green_face"):
            return TLColor.green_right
        if traffic_el.HasField(f"signal_upper_right_arrow_green_face"):
            return TLColor.green_right

        return TLColor.unknown

    def tf_face_color(self, tl_id) -> TLColor:
        if tl_id not in self.tf_face_colors:
            proto_API = self.rasterizer.sat_rast.proto_API
            # tl_colour = TLColor.unknown
            # if proto_API.is_traffic_face_colour(tl_id, "red"):
            #     tl_colour = TLColor.red
            # elif proto_API.is_traffic_face_colour(tl_id, "green"):
            #     tl_colour = TLColor.green
            # elif proto_API.is_traffic_face_colour(tl_id, "yellow"):
            #     tl_colour = TLColor.yellow
            # self.tf_face_colors[tl_id] = tl_colour
            self.tf_face_colors[tl_id] = self.tl_element_color(proto_API[tl_id])

        return self.tf_face_colors[tl_id]

    def __getitem__(self, item_idx):
        fn = self.all_files[item_idx]
        data = np.load(fn, allow_pickle=True)

        agent_id = data["agent_id"].item()
        # non_masked_frame_agents = data["non_masked_frame_agents"].item()
        # agent_data, agent_state = fix_agent_state(*non_masked_frame_agents[agent_id])
        # state_vec = generate_state_vec(self.cfg, data, agent_data, agent_state, item_idx, agent_id)
        agent_from_world = data["agent_from_world"]

        state_vec = None

        if "tl_lanes_masks4" in data:
            image = np.concatenate((data["image_box"], data["image_semantic"]), axis=2).transpose(2, 0, 1)
            image = image.astype(np.float32) / 255.0

            if self.cfg["model_params"].get("nb_raster4_channels", 0) > 0:
                tl_masks = data["tl_lanes_masks4"].item()
                tl_history = data["history_tl_faces"]
                tl4_steps = 3  # now, 1 sec ago, 2 sec ago
                nb_tl4_colors = 9
                nb_tl4_colors_categories = 2  # known off, known on
                image_tl4 = np.zeros(
                    (tl4_steps * nb_tl4_colors * nb_tl4_colors_categories, image.shape[1] // 4, image.shape[2] // 4),
                    dtype=np.float32)
                for tl_delay_id, tl_delay_frame in enumerate([0, 10, 20]):
                    if tl_delay_frame >= len(tl_history):
                        break
                    for known_tl in tl_history[tl_delay_frame]:
                        face_id = known_tl['face_id']
                        traffic_light_face_status = known_tl['traffic_light_face_status']
                        if traffic_light_face_status[
                            2] < 0.5 and face_id in tl_masks:  # skip unknown and outside of raster
                            color_code = self.tf_face_color(face_id)
                            if color_code != TLColor.unknown:
                                output_plane_id = (
                                                              tl_delay_id * nb_tl4_colors + color_code.value) * nb_tl4_colors_categories
                                # output_value = traffic_light_face_status[0] - 0.2 * traffic_light_face_status[1]
                                image_tl4[output_plane_id, tl_masks[face_id]] = traffic_light_face_status[0]
                                image_tl4[output_plane_id + 1, tl_masks[face_id]] = traffic_light_face_status[1]
            else:
                image_tl4 = np.array([0.0])
        else:
            image = data["image"].astype(np.float32) / 255.0
            image_tl4 = np.array([0.0])

        fn_relative = os.path.relpath(fn, self.root_dir_name)

        other_agents_masks = None

        res = {
            k: data[k]
            for k in [
                "target_availabilities",
                "target_positions",
                "world_from_agent",
                "world_to_image",
                "raster_from_world",
                "raster_from_agent",
                "agent_from_world",
                "centroid",
                "timestamp",
                "track_id",
            ]
        }
        res["item_idx"] = item_idx

        # res["image_blocks_positions_agent"] = image_blocks_positions_agent.astype(np.float32)
        # res["corners"] = corners
        res["fn"] = fn
        res["fn_rel"] = fn_relative
        res["image"] = image
        res["image_4x"] = image_tl4

        if other_agents_masks is not None:
            res["other_agents_masks"] = other_agents_masks

        return res


def build_dataset(cfg, stage):
    """
    Build dataset.
    if several datasets are defined
    in the dict cfg.*_data_loader.datasets then create ConcatDataset
    """
    assert stage in ["train", "val", "test"]
    key = LyftDataset.name_2_dataloader_key[stage]
    cfg = cfg.copy()
    dset_cfg = cfg[key]

    if "datasets" in dset_cfg:
        datasets = []
        for dset_name, params in dset_cfg.datasets.items():
            cur_cfg = cfg.copy()
            # we take only the subconfig with the corresponding name!
            OmegaConf.set_struct(cur_cfg, False)
            cur_cfg[key].update(params)
            OmegaConf.set_struct(cur_cfg, True)
            if cur_cfg[key].prerendered:
                dset_class = LyftDatasetPrerendered
            else:
                dset_class = LyftDataset
            datasets.append(dset_class(dset_name, cfg_data=cur_cfg))
        if len(datasets) > 1:
            return ConcatDataset(datasets)
        else:
            return datasets[0]
    else:
        if dset_cfg.prerendered:
            dset_class = LyftDatasetPrerendered
        else:
            dset_class = LyftDataset
        return dset_class(dset_cfg.dset_name, cfg_data=cfg)


def uncompress_zar(fn_src, fn_dst):
    print(fn_src)
    print(fn_dst)
    print(zarr.storage.default_compressor)
    zarr.storage.default_compressor = None
    ds = ChunkedDataset(fn_src).open(cached=False)

    dst_dataset = ChunkedDataset(fn_dst)
    dst_dataset.initialize()
    #     'w',
    #     # num_scenes=len(ds.scenes),
    #     # num_frames=len(ds.frames),
    #     # num_agents=len(ds.agents),
    #     # num_tl_faces=len(ds.tl_faces)
    # )

    with utils.timeit_context("copy scenes"):
        dst_dataset.scenes.append(ds.scenes[:])
    with utils.timeit_context("copy frames"):
        dst_dataset.frames.append(ds.frames[:])
    with utils.timeit_context("copy agents"):
        for i in tqdm(range(0, len(ds.agents), 1024 * 1024)):
            dst_dataset.agents.append(ds.agents[i: i + 1024 * 1024])
    with utils.timeit_context("copy tl_faces"):
        dst_dataset.tl_faces.append(ds.tl_faces[:])


if __name__ == "__main__":
    # uncompress_zar(f'{config.L5KIT_DATA_FOLDER}/scenes/sample.zarr', f'{config.L5KIT_DATA_FOLDER}/scenes/sample_uncompressed.zarr')
    # uncompress_zar(f'{config.L5KIT_DATA_FOLDER}/scenes/validate.zarr',
    #               f'{config.L5KIT_DATA_FOLDER}/scenes/validate_uncompressed.zarr')
    # uncompress_zar(f'{config.L5KIT_DATA_FOLDER}/scenes/validate_chopped_100/validate.zarr',
    #               f'{config.L5KIT_DATA_FOLDER}/scenes/validate_chopped_100/validate_uncompressed.zarr')
    # uncompress_zar(f'{config.L5KIT_DATA_FOLDER}/scenes/train.zarr',
    #               f'{config.L5KIT_DATA_FOLDER}/scenes/train_uncompressed.zarr')
    pass
