import os

import matplotlib.pyplot as plt
import numpy as np
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from tqdm import tqdm

import config
import utils
from dataset import LyftDataset
from dataset import LyftDatasetPrerendered

os.environ["L5KIT_DATA_FOLDER"] = config.L5KIT_DATA_FOLDER


def get_dataset_cfg(scene_name, map_type="box_semantic_fast", prerendered=False):
    cfg_data = dict(
        raster_params=dict(
            raster_size=[224, 224],
            pixel_size=[0.5, 0.5],
            ego_center=[0.25, 0.5],
            map_type=map_type,
            satellite_map_key="aerial_map/aerial_map.png",
            semantic_map_key="semantic_map/semantic_map.pb",
            dataset_meta_key="meta.json",
            filter_agents_threshold=0.5,
            disable_traffic_light_faces=False,
            pre_render_cache_dir="pre_render_224_0.5_tl",
            segmentation_output=None,  # "simple_4x"
        ),
        train_data_loader=dict(key=f"scenes/{scene_name}.zarr", prerendered=prerendered),
        val_data_loader=dict(key=f"scenes/{scene_name}.zarr"),
        test_data_loader=dict(key=f"scenes/{scene_name}.zarr"),
        model_params=dict(
            history_num_frames=50,  # used to retrive appropriate slices of history data from dataset, but only indices form history_box_frames will be rendered by box_semantic_fast rasterizer
            history_step_size=1,
            history_delta_time=0.1,
            history_box_frames=list(range(51)),  # [0, 1, 2, 4, 8],
            future_num_frames=50,
            future_step_size=1,
            future_delta_time=0.1,
            add_agent_state=True,
            add_agent_state_history=True,
            agent_state_history_steps=8,
            max_agent_in_state_history=16
        ),
    )
    return cfg_data


def check_performance(dataset, name="", num_samples=64 * 20, random_order=False):
    with utils.timeit_context(f"iterate {name} dataset"):
        sample = dataset[63]
        # print("image shape", sample["image"]["image_sem"].shape, sample["image"]["image_sem"].dtype)
        print("Keys:", sample.keys())

        target_positions = sample["target_positions"]
        target_positions_world = transform_points(target_positions, sample["world_from_agent"])
        # output_mask = sample["output_mask"]

        img = dataset.rasterizer.to_rgb(sample["image"].transpose(1, 2, 0))
        plt.imshow(img)

        agents_history = sample["agents_history"]
        cur_frame_positions = agents_history[-1, :, :2] * 100.0
        cur_frame_velocity = agents_history[-1, :, 2:4] * 10.0
        cur_frame_positions_img = transform_points(cur_frame_positions, sample["raster_from_agent"])
        plt.scatter(cur_frame_positions_img[:, 0], cur_frame_positions_img[:, 1])

        plt.scatter(cur_frame_positions_img[:, 0] + cur_frame_velocity[:, 0] * 1.0,
                    cur_frame_positions_img[:, 1] + cur_frame_velocity[:, 1] * 1.0,
                    c='red')

        plt.show()

        nb_samples = len(dataset)
        for i in tqdm(range(num_samples)):
            if random_order:
                sample = dataset[np.random.randint(0, nb_samples)]
            else:
                sample = dataset[i]
            target_positions = sample["target_positions"]
            # target_positions_img_m = sample['target_positions_img_m']

            # print(np.linalg.norm(target_positions[0] - target_positions[-1]),
            #       np.linalg.norm(target_positions_img_m[0] - target_positions_img_m[-1]))

            # plt.imshow(sample['image'][-1, :, :])
            # plt.imshow(np.max(sample['output_mask'], axis=0), alpha=0.5)
            # plt.show()


def check_performance_default(num_samples=64 * 20):
    """
    Defautl datset from l5kit w/o any optimizations
    """
    scene_name = "train"
    cfg_data = get_dataset_cfg(scene_name=scene_name, map_type="py_semantic")
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg_data, dm)
    zarr_dataset = ChunkedDataset(dm.require(f"scenes/{scene_name}.zarr")).open()
    dataset = AgentDataset(cfg_data, zarr_dataset, rasterizer)
    check_performance(dataset, "default")


def check_performance_uncompressed(num_samples=64 * 20):
    scene_name = "train_uncompressed"
    dataset = LyftDataset(
        stage=LyftDataset.STAGE_TRAIN,
        cfg_data=get_dataset_cfg(scene_name=scene_name, map_type="box_semantic_fast"),
    )
    check_performance(dataset, "uncompressed+numba")


def check_performance_prerendered(num_samples=128 * 20, random_order=True):
    # map_type='semantic_debug',
    # map_type='semantic_fast',
    # map_type='box_fast',
    # map_type='box_fast',
    dset_name = "train"
    cfg = get_dataset_cfg(scene_name=dset_name, map_type="box_semantic_fast", prerendered=True)
    if dset_name == "train_XXL":
        cfg["raster_params"]["pre_render_cache_dir"] = "pre_render_h01248_XXL"
        cfg["train_data_loader"]["filepaths_cache"] = "filepaths_1_of_8"

    dataset = LyftDatasetPrerendered(dset_name=dset_name, cfg_data=cfg)
    check_performance(dataset, "prerendered", random_order=random_order)


if __name__ == "__main__":
    check_performance_prerendered()
    # check_performance_uncompressed()
    # check_performance_default()
