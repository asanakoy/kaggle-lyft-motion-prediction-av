import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numba
import numpy as np
from l5kit.data.filter import filter_agents_by_labels, filter_agents_by_track_id
from l5kit.data.zarr_dataset import AGENT_DTYPE
from l5kit.geometry import rotation33_as_yaw, transform_point
from l5kit.rasterization import build_rasterizer, SemanticRasterizer, RenderContext
from l5kit.rasterization.rasterizer import (
    EGO_EXTENT_HEIGHT,
    EGO_EXTENT_LENGTH,
    EGO_EXTENT_WIDTH,
    Rasterizer,
)
from l5kit.rasterization.semantic_rasterizer import (
    filter_tl_faces_by_status,
    elements_within_bounds,
    MapAPI,
    CV2_SHIFT,
    CV2_SHIFT_VALUE,
)

import config

os.environ["L5KIT_DATA_FOLDER"] = config.L5KIT_DATA_FOLDER


@numba.njit()
def transform_points_fast_with_cv2_shift(points: np.ndarray, transf_matrix: np.ndarray):
    res = np.zeros((points.shape[0], 2), dtype=np.int32)
    nb_points: int = points.shape[0]
    m00: float = transf_matrix[0, 0]
    m01: float = transf_matrix[0, 1]
    m02: float = transf_matrix[0, 2]
    m10: float = transf_matrix[1, 0]
    m11: float = transf_matrix[1, 1]
    m12: float = transf_matrix[1, 2]

    for i in range(nb_points):
        p0 = points[i, 0]
        p1 = points[i, 1]
        res[i, 0] = int((p0 * m00 + p1 * m01 + m02) * CV2_SHIFT_VALUE)
        res[i, 1] = int((p0 * m10 + p1 * m11 + m12) * CV2_SHIFT_VALUE)

    return res


@numba.njit()
def transform_points_fast(points: np.ndarray, transf_matrix: np.ndarray):
    res = np.zeros((points.shape[0], 2), dtype=np.float32)
    nb_points: int = points.shape[0]
    m00: float = transf_matrix[0, 0]
    m01: float = transf_matrix[0, 1]
    m02: float = transf_matrix[0, 2]
    m10: float = transf_matrix[1, 0]
    m11: float = transf_matrix[1, 1]
    m12: float = transf_matrix[1, 2]

    for i in range(nb_points):
        p0 = points[i, 0]
        p1 = points[i, 1]
        res[i, 0] = p0 * m00 + p1 * m01 + m02
        res[i, 1] = p0 * m10 + p1 * m11 + m12

    return res


@numba.njit()
def fill_corners(extend, yaw, centriod, transf_matrix, res_idx, res):
    e1: float = extend[0] / 2
    e2: float = extend[1] / 2
    y_s: float = np.sin(yaw)
    y_c: float = np.cos(yaw)

    cx = centriod[0]
    cy = centriod[1]

    points = [[-e1, -e2], [-e1, e2], [e1, e2], [e1, -e2]]

    m00: float = transf_matrix[0, 0]
    m01: float = transf_matrix[0, 1]
    m02: float = transf_matrix[0, 2]
    m10: float = transf_matrix[1, 0]
    m11: float = transf_matrix[1, 1]
    m12: float = transf_matrix[1, 2]

    for i in range(4):
        p0 = points[i][0]
        p1 = points[i][1]

        p0_r = p0 * y_c - p1 * y_s + cx
        p1_r = p0 * y_s + p1 * y_c + cy

        res[res_idx, i, 0] = int((p0_r * m00 + p1_r * m01 + m02) * CV2_SHIFT_VALUE)
        res[res_idx, i, 1] = int((p0_r * m10 + p1_r * m11 + m12) * CV2_SHIFT_VALUE)


def get_ego_as_agent(
    frame: np.ndarray,
) -> np.ndarray:  # TODO this can be useful to have around
    """
    Get a valid agent with information from the frame AV. Ford Fusion extent is used

    Args:
        frame (np.ndarray): the frame we're interested in

    Returns: an agent np.ndarray of the AV

    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent


@numba.njit()
def get_agent_coords(agents, raster_from_world):
    nb_agents: np.int64 = agents.shape[0]
    box_raster_coords = np.zeros((nb_agents, 4, 2), dtype=np.int64)

    for idx in range(nb_agents):
        agent = agents[idx]
        fill_corners(
            extend=agent["extent"],
            yaw=agent["yaw"],
            centriod=agent["centroid"],
            transf_matrix=raster_from_world,
            res_idx=idx,
            res=box_raster_coords,
        )
    return box_raster_coords


def draw_boxes(
    raster_size: Tuple[int, int],
    raster_from_world: np.ndarray,
    agents: np.ndarray,
    color: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected in the image plane.
    Finally, cv2 draws the boxes.

    Args:
        raster_size (Tuple[int, int]): Desired output image size
        world_to_image_space (np.ndarray): 3x3 matrix to convert from world to image coordinated
        agents (np.ndarray): array of agents to be drawn
        color (Union[int, Tuple[int, int, int]]): single int or RGB color

    Returns:
        np.ndarray: the image with agents rendered. RGB if color RGB, otherwise GRAY
    """
    if isinstance(color, int):
        im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
    else:
        im = np.zeros((raster_size[1], raster_size[0], 3), dtype=np.uint8)

    # box_world_coords = np.zeros((len(agents), 4, 2))
    # corners_base_coords = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    #
    # # compute the corner in world-space (start in origin, rotate and then translate)
    # for idx, agent in enumerate(agents):
    #     corners = corners_base_coords * agent["extent"][:2] / 2  # corners in zero
    #     r_m = yaw_as_rotation33(agent["yaw"])
    #     box_world_coords[idx] = transform_points_fast(corners, r_m) + agent["centroid"][:2]
    #
    # box_raster_coords = transform_points_fast_with_cv2_shift(box_world_coords.reshape((-1, 2)), raster_from_world)
    #
    # # fillPoly wants polys in a sequence with points inside as (x,y)
    # box_raster_coords = box_raster_coords.reshape((-1, 4, 2))

    box_raster_coords = get_agent_coords(agents, raster_from_world)

    cv2.fillPoly(im, box_raster_coords, color=color, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
    return im


class FastBoxRasterizer(Rasterizer):
    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        box_history_frames: List[int],
    ):
        """

        Args:
            render_context (RenderContext): Render context
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
        """
        super(FastBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.filter_agents_threshold = filter_agents_threshold
        # Currently not used
        self.history_num_frames = None  # history_num_frames
        self.box_history_frames = box_history_frames

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        # all frames are drawn relative to this one"
        frame = history_frames[0]
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (
            self.raster_size[1],
            self.raster_size[0],
            len(self.box_history_frames),
        )
        agents_images = np.zeros(out_shape, dtype=np.uint8)
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        # for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
        for dst_idx, i in enumerate(self.box_history_frames):
            if i >= history_frames.shape[0]:
                break
            frame = history_frames[i]
            agents = history_agents[i]

            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:
                agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                ego_image = draw_boxes(self.raster_size, raster_from_world, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if len(agent_ego) == 0:  # agent not in this history frame
                    agents_image = draw_boxes(
                        self.raster_size,
                        raster_from_world,
                        np.append(agents, av_agent),
                        255,
                    )
                    ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    agents = agents[agents != agent_ego[0]]
                    agents_image = draw_boxes(
                        self.raster_size,
                        raster_from_world,
                        np.append(agents, av_agent),
                        255,
                    )
                    ego_image = draw_boxes(self.raster_size, raster_from_world, agent_ego, 255)

            agents_images[..., dst_idx] = agents_image
            ego_images[..., dst_idx] = ego_image

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        out_im = np.concatenate((agents_images, ego_images), -1)

        return {'image_box': out_im}

    def to_rgb(self, in_im_dict: Dict[str, np.ndarray], **kwargs: dict) -> np.ndarray:
        """
        get an rgb image where agents further in the past have faded colors

        Args:
            in_im: the output of the rasterize function
            kwargs: this can be used for additional customization (such as colors)

        Returns: an RGB image with agents and ego coloured with fading colors
        """
        in_im = in_im_dict['image_box']
        hist_frames = in_im.shape[-1] // 2
        in_im = np.transpose(in_im, (2, 0, 1))

        # this is similar to the draw history code
        out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_chs = in_im[:hist_frames][::-1]  # reverse to start from the furthest one
        agent_color = (0, 0, 1) if "agent_color" not in kwargs else kwargs["agent_color"]
        for ch in agent_chs:
            out_im_agent *= 0.85  # magic fading constant for the past
            out_im_agent[ch > 0] = agent_color

        out_im_ego = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        ego_chs = in_im[hist_frames:][::-1]
        ego_color = (0, 1, 0) if "ego_color" not in kwargs else kwargs["ego_color"]
        for ch in ego_chs:
            out_im_ego *= 0.85
            out_im_ego[ch > 0] = ego_color

        out_im = (np.clip(out_im_agent + out_im_ego, 0, 1) * 255).astype(np.uint8)
        return out_im


class FastSemanticRasterizer(SemanticRasterizer):
    def __init__(self, render_context, semantic_map_path: str, world_to_ecef: np.ndarray):
        super().__init__(render_context, semantic_map_path, world_to_ecef)

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        sem_im = self.render_semantic_map_cv(center_in_world_m, raster_from_world, history_tl_faces)
        return sem_im

    def render_semantic_map_cv(
        self, center_world: np.ndarray, world_to_image_space: np.ndarray, history_tl_faces: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_world (np.ndarray): XY of the image center in world ref system
            world_to_image_space (np.ndarray):
        Returns:
            np.ndarray: RGB raster

        """

        tl_faces = history_tl_faces[0]

        img = 255 * np.ones(shape=(self.raster_size[1], self.raster_size[0], 3), dtype=np.uint8)

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())
        all_tl_ids = set(tl_faces["face_id"].tolist())

        # plot lanes
        lanes_lines = defaultdict(list)

        rasterized_tl_lanes: Dict[str, np.ndarray] = {}

        all_tl_ids_from_lanes = set()

        for idx in elements_within_bounds(center_world, self.bounds_info["lanes"]["bounds"], raster_radius):
            lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane
            # 2.1s
            # get image coords
            lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
            # 2.1 s

            # xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], world_to_image_space))
            # xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], world_to_image_space))
            # 5.5 s
            xy_left = transform_points_fast_with_cv2_shift(lane_coords["xyz_left"], world_to_image_space)
            xy_right = transform_points_fast_with_cv2_shift(lane_coords["xyz_right"], world_to_image_space)
            # 2.7s

            lanes_area = np.vstack((xy_left, np.flip(xy_right, 0)))  # start->end left then end->start right
            # 5.7s -> 3.5s

            # Note(lberg): this called on all polygons skips some of them, don't know why
            cv2.fillPoly(img, [lanes_area], (17, 17, 31), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
            # 7s

            lane_type = "default"  # no traffic light face is controlling this lane
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            all_tl_ids_from_lanes.update(lane_tl_ids)
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"

            for tl_id in lane_tl_ids:
                if tl_id not in rasterized_tl_lanes:
                    rasterized_tl_lanes[tl_id] = np.zeros(shape=(self.raster_size[1]//4,
                                                                 self.raster_size[0]//4), dtype=np.uint8)
                cv2.fillPoly(rasterized_tl_lanes[tl_id], [lanes_area], 1, lineType=cv2.LINE_4, shift=CV2_SHIFT+2)
                # cv2.polylines(
                #     rasterized_tl_lines[tl_id], [xy_left, xy_right], False, 1, lineType=cv2.LINE_4, shift=CV2_SHIFT,
                # )

            lanes_lines[lane_type].extend([xy_left, xy_right])
            # 7.4 s

        cv2.polylines(
            img,
            lanes_lines["default"],
            False,
            (255, 217, 82),
            lineType=cv2.LINE_AA,
            shift=CV2_SHIFT,
        )
        cv2.polylines(
            img,
            lanes_lines["green"],
            False,
            (0, 255, 0),
            lineType=cv2.LINE_AA,
            shift=CV2_SHIFT,
        )
        cv2.polylines(
            img,
            lanes_lines["yellow"],
            False,
            (255, 255, 0),
            lineType=cv2.LINE_AA,
            shift=CV2_SHIFT,
        )
        cv2.polylines(
            img,
            lanes_lines["red"],
            False,
            (255, 0, 0),
            lineType=cv2.LINE_AA,
            shift=CV2_SHIFT,
        )

        # only for testing, positions of TL
        # tl_positions_img = np.zeros(shape=(self.raster_size[1], self.raster_size[0]), dtype=np.uint8)
        # for tl_id in list(all_tl_ids_from_lanes):
        #     e = self.proto_API[tl_id].element.traffic_control_element
        #     xy = self.proto_API.unpack_deltas_cm(
        #         e.points_x_deltas_cm,
        #         e.points_y_deltas_cm,
        #         e.points_z_deltas_cm,
        #         e.geo_frame)
        #     xy = transform_points_fast_with_cv2_shift(xy, world_to_image_space)
        #     print(xy.mean(axis=0) / CV2_SHIFT_VALUE)
        #     pos = (xy.mean(axis=0) / CV2_SHIFT_VALUE).astype(np.int)
        #     if pos.min() > 8 and pos.max() < tl_positions_img.shape[0] - 8:
        #         tl_positions_img[pos[1] - 8:pos[1] + 8, pos[0] - 8:pos[0] + 8] += 1

        # 8.0s

        # plot crosswalks
        # prepare: 0.2s
        crosswalks = []
        for idx in elements_within_bounds(center_world, self.bounds_info["crosswalks"]["bounds"], raster_radius):
            crosswalk = self.proto_API.get_crosswalk_coords(self.bounds_info["crosswalks"]["ids"][idx])

            xy_cross = transform_points_fast_with_cv2_shift(crosswalk["xyz"], world_to_image_space)
            crosswalks.append(xy_cross)

        cv2.polylines(img, crosswalks, True, (255, 117, 69), lineType=cv2.LINE_AA, shift=CV2_SHIFT)

        non_missing_rasterized_tl_lanes = {tl_face: mask.astype(np.bool)
                                           for tl_face, mask in rasterized_tl_lanes.items()
                                           if np.any(mask)}

        return {'image_semantic': img, 'tl_lanes_masks4': non_missing_rasterized_tl_lanes}

    def to_rgb(self, in_im_dict: dict, **kwargs: dict) -> np.ndarray:
        return in_im_dict['image_semantic']


def _load_metadata(meta_key: str, data_manager) -> dict:
    """
    Load a json metadata file

    Args:
        meta_key (str): relative key to the metadata
        data_manager (DataManager): DataManager used for requiring files

    Returns:
        dict: metadata as a dict
    """
    metadata_path = data_manager.require(meta_key)
    with open(metadata_path, "r") as f:
        metadata: dict = json.load(f)
    return metadata


def get_hardcoded_world_to_ecef() -> np.ndarray:  # TODO remove when new dataset version is available
    """
    Return and hardcoded world_to_ecef matrix for dataset V1.0

    Returns:
        np.ndarray: 4x4 matrix
    """
    print(
        "!!dataset metafile not found!! the hard-coded matrix will be loaded.\n"
        "This will be deprecated in future releases"
    )

    world_to_ecef = np.asarray(
        [
            [8.46617444e-01, 3.23463078e-01, -4.22623402e-01, -2.69876744e06],
            [-5.32201938e-01, 5.14559352e-01, -6.72301845e-01, -4.29315158e06],
            [-3.05311332e-16, 7.94103464e-01, 6.07782600e-01, 3.85516476e06],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ],
        dtype=np.float64,
    )
    return world_to_ecef


class FastSemBoxRasterizer(Rasterizer):
    """Combine a Semantic Map and a Box Rasterizers into a single class"""

    def __init__(
        self,
        render_context: RenderContext,
        filter_agents_threshold: float,
        history_num_frames: int,
        semantic_map_path: str,
        world_to_ecef: np.ndarray,
        box_history_frames: List[int],
    ):
        super(FastSemBoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.pixel_size = render_context.pixel_size_m
        self.ego_center = render_context.center_in_raster_ratio
        self.filter_agents_threshold = filter_agents_threshold
        # Not used, use box_history_frames instead
        self.history_num_frames = None  # history_num_frames

        self.box_rast = FastBoxRasterizer(
            render_context,
            filter_agents_threshold,
            history_num_frames,
            box_history_frames,
        )
        self.sat_rast = FastSemanticRasterizer(render_context, semantic_map_path, world_to_ecef)

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        im_out_box = self.box_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        im_out_sat = self.sat_rast.rasterize(history_frames, history_agents, history_tl_faces, agent)
        return {
            **im_out_sat,
            **im_out_box,
            'history_frames': history_frames,
            'history_agents': history_agents,
            'history_tl_faces': history_tl_faces
        }
        # return np.concatenate([im_out_box, im_out_sat], -1)

    def to_rgb(self, in_im, **kwargs: dict) -> np.ndarray:
        if isinstance(in_im, dict):
            im_out_box = self.box_rast.to_rgb(in_im, **kwargs)
            im_out_sat = self.sat_rast.to_rgb(in_im, **kwargs)
        else:
            im_out_box = self.box_rast.to_rgb({'image_box': in_im[..., :-3]}, **kwargs)
            im_out_sat = self.sat_rast.to_rgb({'image_semantic': in_im[..., -3:]}, **kwargs)
        # merge the two together
        mask_box = np.any(im_out_box > 0, -1)
        im_out_sat[mask_box] = im_out_box[mask_box]
        return im_out_sat


def build_custom_rasterizer(cfg: dict, data_manager):
    raster_cfg = cfg["raster_params"]
    map_type = raster_cfg["map_type"]

    if map_type not in ("semantic_fast", "box_fast", "box_semantic_fast"):
        return build_rasterizer(cfg, data_manager)

    raster_cfg = cfg["raster_params"]
    map_type = raster_cfg["map_type"]
    dataset_meta_key = raster_cfg["dataset_meta_key"]

    render_context = RenderContext(
        raster_size_px=np.array(raster_cfg["raster_size"]),
        pixel_size_m=np.array(raster_cfg["pixel_size"]),
        center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
    )

    filter_agents_threshold = raster_cfg["filter_agents_threshold"]
    history_num_frames = cfg["model_params"]["history_num_frames"]
    history_box_frames = cfg["model_params"].get("history_box_frames", [])

    semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
    try:
        dataset_meta = _load_metadata(dataset_meta_key, data_manager)
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
    except (
        KeyError,
        FileNotFoundError,
    ):  # TODO remove when new dataset version is available
        world_to_ecef = get_hardcoded_world_to_ecef()

    if map_type == "semantic_fast":
        return FastSemanticRasterizer(render_context, semantic_map_filepath, world_to_ecef)
    elif map_type == "box_fast":
        return FastBoxRasterizer(
            render_context,
            filter_agents_threshold,
            history_num_frames,
            history_box_frames,
        )
    elif map_type == "box_semantic_fast":
        return FastSemBoxRasterizer(
            render_context=render_context,
            filter_agents_threshold=filter_agents_threshold,
            world_to_ecef=world_to_ecef,
            semantic_map_path=semantic_map_filepath,
            history_num_frames=history_num_frames,
            box_history_frames=history_box_frames,
        )
