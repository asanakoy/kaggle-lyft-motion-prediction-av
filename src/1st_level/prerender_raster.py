import argparse

import os
import numpy as np
import multiprocessing
from tqdm import tqdm
import numcodecs
import glob

from l5kit.data import LocalDataManager

import config
import dataset
import utils


numcodecs.blosc.set_nthreads(1)

os.environ["L5KIT_DATA_FOLDER"] = config.L5KIT_DATA_FOLDER
DST_DIR = None  # f"{config.L5KIT_DATA_FOLDER}/pre_render_320_0.3"

"""
pre_render_512_0.3:
                raster_size=[512, 512],
                pixel_size=[0.3, 0.3],
                ego_center=[0.15625, 0.5]


pre_render_224_0.5:
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5]

pre_render_h01248: default
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5],
                history_box_frames=[0, 1, 2, 4, 8],

pre_render_h50all: default
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5],
                history_box_frames=range(0, 51),
pre_render_320_0.3:
                raster_size=[320, 320],
                pixel_size=[0.3, 0.3],
                ego_center=[0.25, 0.5],

pre_render_256_0.5:
                raster_size=[256, 256],
                pixel_size=[0.5, 0.5],
                ego_center=[0.1875, 0.5],  # keep 0 pos at the center of 32x32 block for better pos encoding

pre_render_288_0.3:
                raster_size=[288, 288],
                pixel_size=[0.3, 0.3],
                ego_center=[0.1666666666666667, 0.5],

pre_render_288_0.5:
                raster_size=[288, 288],
                pixel_size=[0.5, 0.5],
                ego_center=[0.1666666666666667, 0.5],
"""


def create_lyft_dataset(dset_name, zarr_name):
    ds = dataset.LyftDataset(
        dset_name=dset_name,
        cfg_data=dict(
            raster_params=dict(
                raster_size=[224, 224],
                pixel_size=[0.5, 0.5],
                ego_center=[0.25, 0.5],
                map_type="box_semantic_fast",
                satellite_map_key="aerial_map/aerial_map.png",
                semantic_map_key="semantic_map/semantic_map.pb",
                dataset_meta_key="meta.json",
                filter_agents_threshold=0.5,
                disable_traffic_light_faces=False,
            ),
            train_data_loader=dict(key=f"scenes/{zarr_name}.zarr"),
            val_data_loader=dict(key=f"scenes/{zarr_name}.zarr"),
            test_data_loader=dict(key=f"scenes/{zarr_name}.zarr"),
            model_params=dict(
                history_num_frames=20,  # used to retrive appropriate slices of history data from dataset, but only indices form history_box_frames will be rendered by box_semantic_fast rasterizer
                history_step_size=1,
                history_delta_time=0.1,
                history_box_frames=[0, 1, 2, 4, 8],
                future_num_frames=50,
                future_step_size=1,
                future_delta_time=0.1,
                add_agent_state=False,
            ),
        ),
    )
    return ds


def pos_ahead(agent):
    time_ahead = 2.5
    distance_ahead = 5.0

    xy = agent["centroid"]
    vel = agent["velocity"]
    return xy + vel * time_ahead


def filter_agents(history_agents, current_agent):
    current_agents = history_agents[0]
    agent_pos_ahead = pos_ahead(current_agent)

    current_agents_sorted = list(
        sorted(
            current_agents,
            key=lambda x: np.linalg.norm(agent_pos_ahead - pos_ahead(x)),
        )
    )
    return history_agents


def filter_tl_faces(history_tl_faces):
    return [
        t[t['traffic_light_face_status'][:, 2] < 0.5]
        for t in history_tl_faces
    ]


def pre_render_scenes(initial_scene, dset_name, scene_step, zarr_name, skip_frame_step, verbose=10):
    print(f"Job {initial_scene}: creating dataset...")
    with utils.timeit_context(f"Job {initial_scene} dataset creation"):
        ds = create_lyft_dataset(dset_name, zarr_name)
    print(f"Job {initial_scene}: creating dataset... [OK]")

    nb_scenes = len(ds.zarr_dataset.scenes)
    if verbose:
        print("total scenes:", nb_scenes)

    for scene_num in tqdm(
        range(initial_scene, nb_scenes, scene_step),
        desc=f"job {initial_scene} scenes:",
        disable=not verbose,
        total=(nb_scenes - initial_scene) // scene_step,
    ):
        if verbose >= 10:
            print("processing scene", scene_num)
        from_frame, to_frame = ds.zarr_dataset.scenes["frame_index_interval"][scene_num]
        all_frames = ds.zarr_dataset.frames[from_frame:to_frame].copy()

        for frame_num in range(from_frame, to_frame):
            if (frame_num + 1) % (1 + skip_frame_step) > 0:
                continue

            dir_created = False
            state_index = frame_num - from_frame
            dst_dir = f"{DST_DIR}/{zarr_name}/{scene_num:05}/{state_index:03}/"

            agent_from, agent_to = all_frames["agent_index_interval"][state_index]

            frame_agents = ds.zarr_dataset.agents[agent_from:agent_to]
            # frame_agents_state = ds.agent_state[agent_from:agent_to]
            relative_agents_indices = np.nonzero(ds.agent_dataset.agents_mask[agent_from:agent_to])[0]

            non_masked_agents = {}
            for agent_num in relative_agents_indices:
                non_masked_agents[agent_num + agent_from] = (
                    frame_agents[agent_num],
                    None,  # frame_agents_state[agent_num],
                )

            data_by_agent = {}
            target_by_agent_id = {}

            for agent_num in relative_agents_indices:
                track_id = frame_agents[agent_num]["track_id"]

                data = ds.agent_dataset.get_frame(scene_num, state_index=state_index, track_id=track_id)
                raster = data["image"]

                del data["image"]
                # data["image"] = img  # np.clip(img * 255.0, 0, 255).astype(np.uint8)
                data["image_semantic"] = raster["image_semantic"]
                data["image_box"] = raster["image_box"]
                data["tl_lanes_masks4"] = raster["tl_lanes_masks4"]

                # data["history_frames"] = raster["history_frames"]
                # data["history_agents"] = filter_agents(raster["history_agents"], frame_agents[agent_num])
                data["history_tl_faces"] = filter_tl_faces(raster["history_tl_faces"])

                # print(img.shape)
                # agent_state = frame_agents_state[agent_num]
                data["agent_state"] = frame_agents[agent_num]
                data["agent_id"] = agent_num + agent_from
                data["scene_id"] = scene_num
                data["frame_id"] = frame_num
                data["non_masked_frame_agents"] = non_masked_agents
                # warning all_frame_agent is very large and will sacriface the reading speed
                # data["all_frame_agents"] = frame_agents

                data_by_agent[track_id] = data
                target_by_agent_id[track_id] = {
                    key: data[key]
                    for key in [
                        "target_availabilities",
                        "target_positions",
                        "world_from_agent",
                        "world_to_image",
                        "raster_from_world",
                        "raster_from_agent",
                        "agent_from_world",
                    ]
                }

            for agent_num in relative_agents_indices:
                track_id = frame_agents[agent_num]["track_id"]
                fn = f"{dst_dir}/{track_id:04}.npz"
                # if os.path.exists(fn):
                # if verbose >= 1:
                #    print(f" - {fn} is already rendered")
                # continue

                if not dir_created:
                    os.makedirs(dst_dir, exist_ok=True)
                    dir_created = True

                data = data_by_agent[track_id]
                np.savez_compressed(fn, **data, target_by_agent_id=target_by_agent_id)


def pre_render_parallel(dset_name, scene_step, zarr_name, skip_frame_step, initial_scenes, num_jobs):
    # if zarr_name in ["train_uncompressed", "validate_uncompressed"] and
    if dset_name in [
        "train",
        "train_XXL",
        "val",
    ]:
        try:
            # Agent masks must be generated first
            dm = LocalDataManager(None)
            dm.require(f"scenes/{zarr_name}.zarr/agents_mask/0.5/.zarray")
        except FileNotFoundError:
            print("-- Create dataset to generate agent masks first")
            _ = create_lyft_dataset(dset_name, zarr_name)

    p = multiprocessing.Pool(num_jobs)
    res = []
    for i, initial_scene in enumerate(initial_scenes):
        res.append(
            p.apply_async(
                pre_render_scenes,
                kwds=dict(
                    initial_scene=initial_scene,
                    scene_step=scene_step,
                    zarr_name=zarr_name,
                    skip_frame_step=skip_frame_step,
                    dset_name=dset_name,
                    verbose=(i == 0),
                ),
            )
        )
        # pre_render_scenes(
        #     initial_scene=initial_scene,
        #     scene_step=scene_step,
        #     zarr_name=zarr_name,
        #     skip_frame_step=skip_frame_step,
        #     dset_name=dset_name,
        #     verbose=(i == 0),
        # )

    for r in res:
        print(".")
        r.get()
    print("Done all")

    dataset_dir = f"{DST_DIR}/{zarr_name}"
    all_files_fn = f"{dataset_dir}/filepaths_1_of_{skip_frame_step + 1}.npy"
    print(f"Generating and caching filenames in {all_files_fn}")
    all_files = list(sorted(glob.glob(f"{dataset_dir}/**/*.npz", recursive=True)))
    np.save(all_files_fn, all_files)
    print(f"Generated all npz paths and saved to {all_files_fn}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="render")
    parser.add_argument("--dset_name", type=str, default="train")
    parser.add_argument("--zarr_name", type=str, default=None)
    parser.add_argument("--dir_name", type=str, default="pre_render_h01248")
    # parser.add_argument('--initial_scene', type=int, default=0)
    parser.add_argument("--scene_step", type=int, default=1)
    parser.add_argument("--skip_frame_step", type=int, default=0)
    parser.add_argument("--initial_scenes", type=int, nargs="+")
    parser.add_argument("--num_jobs", type=int, default=16)

    args = parser.parse_args()
    DST_DIR = f"{config.L5KIT_DATA_FOLDER}/{args.dir_name}"
    print("Root DST DIR:", DST_DIR)
    action = args.action
    dset_name = args.dset_name

    zarr_names = {
        "train_XXL": "train_XXL",
        "train": "train",
        "val": "validate",
        "test": "test",
    }
    zarr_name = zarr_names[dset_name] if args.zarr_name is None else args.zarr_name

    if action == "render":
        pre_render_parallel(
            scene_step=args.scene_step,
            zarr_name=zarr_name,
            dset_name=dset_name,
            skip_frame_step=args.skip_frame_step,
            initial_scenes=args.initial_scenes,
            num_jobs=args.num_jobs,
        )
