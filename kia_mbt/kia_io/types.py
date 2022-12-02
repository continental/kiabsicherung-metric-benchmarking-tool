# Copyright (c) 2022 Continental AG and subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains custom types for the KIA dataset loader.

"""

from typing import List
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


@dataclass
class KIADatasetConfig(object):
    """
    Data class for the configuration of the KIA Dataset Loader

    Parameters
    ----------
        sequences : List[int]
            List containing sequence numbers, which will be loaded. If empty all
            sequences will be loaded.

        sequence_names : List[str]
            List containing sequence names, which shall be loaded. If empty all
            sequences will be loaded. This one is prefered over the list of
            sequences with numbers.

        company : str
            Defines if only data from one data producer shall be loaded. Valid
            is 'bit' for BIT-TS and 'mv' for MackeVision. An empty string means
            that any data is loaded.

        tranches : List[int]
            The tranches that shall be loaded. If empty all tranches will be
            loaded except when the sequences list is not empty.

        dataset_split : str
            The dataset split that shall be loaded, which can be 'train', 'val',
            'test'. When it is empty all splits are loaded except when the
            sequence list is not empty.

        get_image_png : bool
            If this flag is set to True, PNG images will be loaded when data
            loading is accessed via [] (indexer) operator.

        get_image_exr : bool
            If this flag is set to True, EXR images will be loaded when data
            loading is accessed via [] (indexer) operator.

        get_grp_seg : bool
            If this flag is set to True, group segmentation will be loaded when
            data loading is accessed via [] (indexer) operator.

        get_inst_seg : bool
            If this flag is set to True, instance segmentation will be loaded
            when data loading is accessed via [] (indexer) operator.

        get_body_part : bool
            If this flag is set to True, body part segmentation's will be loaded
            when data loading is accessed via [] (indexer) operator.

        get_detections_2d : bool
            If this flag is set to True, 2D bounding box detections will be
            loaded when data loading is accessed via [] (indexer) operator.
    """

    sequences: List[int] = field(default_factory=list)
    sequence_names: List[str] = field(default_factory=list)
    company: str = ""
    tranches: List[int] = field(default_factory=list)
    dataset_split: str = ""
    get_image_png: bool = False
    get_image_exr: bool = False
    get_grp_seg: bool = False
    get_inst_seg: bool = False
    get_body_part: bool = False
    get_detections_2d: bool = True
    get_meta_info: bool = True


@dataclass
class KIADetection2D(object):
    """
    Data class for representing a 2D bounding box detection

    For the detailed documentation of the enriched 2d boundig box annotations,
    please check the page https://confluence.vdali.de/x/QQV.

    Parameters
    ----------
        class_id : str
            Name of the class

        sensor : str
            Type of the used sensor, e.g. camera

        center : np.ndarray
            Coordinate of the bounding box in Pixel space

        size : np.ndarray
            Width and height of the bounding box in Pixels

        rotation : float
            Rotation or orientation of the object in degree

        confidence : float
            confidence score of a detection

        occlusion : float
            Amount of occlusion in degrees, where 0 means no occlusion

        occlusion_estimate : float
            If no ground truth for occlusion is given, this provides an estimate

        velocity : np.ndarray
            Velocity of the object in px/s in x- and y-direction

        truncated : bool
            True if bounding box is truncated, e.g. at the image boarder

        instance_id : int
            Unique identifier for an object

        object_id: int
            Legacy. Same as instance ID

        depth : float
            Relative depth in meters with respect to the camera origin

        instance_pixels : int
            Number of visible pixels of the object

        angle : float
            Angle between the optical axis of the camera (x-axis) and the
            position of the object

        pos_cc : np.ndarray
            Position in camera coordinate system in meters

        pos_bev : np.ndarray
            Position in birds-eye view image coordinates in pixels (u,v)

        within_brake_dist_30kph : bool
            Flag that indicates if object is within 30kph detection zone

        within_brake_dist_50kph : bool
            Flag that indicates if object is within 50kph detection zone

        semantic_area : str
            Ground type on which the object is standing on

        eval_cat_a : bool
            Flag that indicates if object is of category A

        eval_cat_b : bool
            Flag that indicates if object is of category B
    """

    class_id: str = "Unknown"
    sensor: str = ""
    center: np.ndarray = np.full(2, np.nan, int)
    size: np.ndarray = np.full(2, np.nan, int)
    rotation: float = 0.0
    confidence: float = 1.0
    occlusion: float = -1.0
    occlusion_estimate: float = -1.0
    velocity: np.ndarray = np.full(2, np.nan, float)
    truncated: bool = False
    instance_id: int = 0
    object_id: int = 0
    depth: float = -1.0
    instance_pixels: int = -1
    angle: float = None
    pos_cc: np.ndarray = np.full(2, np.nan, float)
    pos_bev: np.ndarray = np.full(2, np.nan, float)
    within_brake_dist_30kph: bool = None
    within_brake_dist_50kph: bool = None
    semantic_area: str = "other"
    eval_cat_a: bool = None
    eval_cat_b: bool = None


@dataclass
class KIALightSourceInformation(object):
    instance_id: str = None
    world_angles_azimuth: float = np.nan
    world_angles_elevation: float = np.nan
    world_sky: str = None
    world_elevation: str = None


@dataclass
class KIAEgoSensorInformation(object):
    instance_id: str = None
    angle_bev_north2fov_deg: float = np.nan


@dataclass
class KIAInstanceSensorInformation(object):
    sensor_id: str = None
    angles_car_cosy_angle_sensor_dir2obj_deg: float = np.nan
    angles_car_cosy_angle_sensor_dir2hip_dir_deg: float = np.nan
    angles_car_cosy_angle_sensor_dir2eyes_dir_deg: float = np.nan
    metainfo_plf_contrast_rgb: float = np.nan
    metainfo_plf_luminance_inst_dyn_range: float = np.nan
    sensor_occlusion_rate: float = np.nan
    sensor_occlusion_type: str = None
    sensor_occlusion_total_pixels: int = -1
    sensor_occlusion_visible_pixels: int = -1
    sensor_occlusion_visible_skeleton_parts_from_joints: List[str] = field(
        default_factory=list
    )


@dataclass
class KIAEntityInformaton(object):
    instance_id: int = -1
    prototype_asset_id: str = None
    prototype_ood_asset: bool = False
    prototype_mocap_asset: bool = False
    world_semantic_area: str = None
    world_position_level_relative_diff_eye_lowest_bodypart: float = np.nan
    instance_sensors: List[KIAInstanceSensorInformation] = field(default_factory=list)


@dataclass
class KIAAdditionalMetaInformation(object):
    """
    Dataclass for storing additional meta information for an instance.

    The additional meta information are contained in the following ground truth
    folder in the KIA dataset:

    general-globally-per-frame-analysis-enriched_json

    Parameters
    ----------
        entities : List[KIAEntityInformaton]
            List of entities meta information.

        light_sources : List[KIALightSourceInformation]
            List of light sources meta information.

        ego_sensors : List[KIAEgoSensorInformation]
            List of ego sensor meta information.
    """

    entities: List[KIAEntityInformaton] = field(default_factory=list)
    light_sources: List[KIALightSourceInformation] = field(default_factory=list)
    ego_sensors: List[KIAEgoSensorInformation] = field(default_factory=list)


@dataclass
class KIADataContainer(object):
    """
    Data container for one data sample.

    Parameters
    ----------
        sample_name : str
            Name of the sample.

        image_png : Image
            Camera image as PNG of the sample.

        image_exr : Image
            Camera image as EXR of the sample.

        grp_seg : Image
            Semantic group segmentation of the sample.

        inst_seg : Image
            Semantic instance segmentation of the sample.

        body_part : Image
            Body part segmentation's of the sample.

        detections_2d : List[KIADetection2D]
            2D bounding box detections of the sample.

        meta_info : KIAAdditionalMetaInformation
            Additional meta information of the sample.
    """

    sample_name: str = ""
    image_png: Image.Image = None
    image_exr: Image.Image = None
    grp_seg: Image.Image = None
    inst_seg: Image.Image = None
    body_part: Image.Image = None
    meta_info: KIAAdditionalMetaInformation = None
    detections_2d: List[KIADetection2D] = field(default_factory=list)


@dataclass
class KIAPredictionContainer(object):
    """
    Prediction container for one data sample

    Parameters
    ----------
        sample_name : str
            Name of the sample

        detections_2d : List[KIADetection2D]
            2D bounding box predictions for the sample
    """

    sample_name: str = ""
    detections_2d: List[KIADetection2D] = field(default_factory=list)
