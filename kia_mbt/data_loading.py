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
# limitations under the License.s
"""
This file contains data loading functions for the MBT.
"""

import os
import sys
import logging
import pandas as pd
import kia_mbt.kia_io as mbt_io
import kia_mbt.config_loader as mbt_config


def get_backend(config: mbt_config.IOConfig, path: str) -> mbt_io.KIADatasetBackend:
    """
    Creates backend from configuration.

    Currently, this function supports the file system and MinIO backend.

    Parameters
    ----------
        config : IOConfig
            IO module configuration.

        path : str
            Data path.

    Returns
    -------
    Backend.
    """

    backend = None
    if config.backend == "fs":
        backend = mbt_io.KIADatasetFSBackend(path)
    elif config.backend == "minio":
        try:
            access_key = os.environ["KIA_MBT_MINIO_ACCESS_KEY"]
            secret_key = os.environ["KIA_MBT_MINIO_SECRET_KEY"]
            backend = mbt_io.KIADatasetMinIOBackend(
                access_key,
                secret_key,
                config.minio_endpoint,
                config.minio_bucket,
                path,
                config.minio_use_proxy,
            )
        except KeyError as key_error_exception:
            logging.error("Missing environment variable: %s", key_error_exception)
            sys.exit()
    else:
        logging.error("No data loading backend named %s available", config.backend)
        sys.exit()

    return backend


def load_2dbb_annotations(
    backend: mbt_io.KIADatasetBackend, config: mbt_io.KIADatasetConfig
) -> pd.DataFrame:
    """
    This function loads 2D bounding box annotations into a data frame.

    Based on the given dataset configuration, the respective data samples will
    be loaded and the 2d bounding box annotations will be stored within a
    dictionary, whoch is finally converted into a data frame. This is done for
    processing time reasons.

    Parameters
    ----------
    backend : KIADatasetBackend
        Backend for the KIA dataset loader
    config : KIADatasetConfig
        KIA dataset configuration

    Returns
    -------
    Data frame with 2d bounding box annotations
    """

    data = {}
    # create KIA dataset loader with backend and config
    data_loader = mbt_io.KIADatasetLoader(backend, config)
    # iterate over all data samples and store them into dictionary
    for i in range(0, len(data_loader)):
        data_sample = data_loader[i]
        for annotation in data_sample.detections_2d:
            # create key entry for dataframe
            key = data_sample.sample_name + "/" + str(annotation.instance_id)
            # search for matching entity by instance id in meta information
            if not data_sample.meta_info:
                data_sample.meta_info = mbt_io.KIAAdditionalMetaInformation()
                data_sample.meta_info.light_sources.append(
                    mbt_io.KIALightSourceInformation()
                )
                data_sample.meta_info.ego_sensors.append(
                    mbt_io.KIAEgoSensorInformation()
                )
            entity = [
                e
                for e in data_sample.meta_info.entities
                if e.instance_id == annotation.instance_id
            ]
            if entity:
                entity = entity[0]
            else:
                entity = mbt_io.KIAEntityInformaton()
                entity.instance_sensors.append(mbt_io.KIAInstanceSensorInformation())

            data[key] = [
                data_sample.sample_name,
                annotation.instance_id,
                annotation.instance_pixels,
                annotation.object_id,
                annotation.occlusion,
                annotation.occlusion_estimate,
                annotation.rotation,
                annotation.sensor,
                annotation.size,
                annotation.truncated,
                annotation.velocity,
                annotation.center,
                annotation.class_id,
                annotation.depth,
                annotation.angle,
                annotation.pos_cc,
                annotation.pos_bev,
                annotation.within_brake_dist_30kph,
                annotation.within_brake_dist_50kph,
                annotation.semantic_area,
                annotation.eval_cat_a,
                annotation.eval_cat_b,
                entity.prototype_asset_id,
                entity.prototype_mocap_asset,
                entity.prototype_ood_asset,
                entity.world_position_level_relative_diff_eye_lowest_bodypart,
                entity.world_semantic_area,
                entity.instance_sensors[0].sensor_id,
                entity.instance_sensors[0].metainfo_plf_luminance_inst_dyn_range,
                entity.instance_sensors[0].metainfo_plf_contrast_rgb,
                entity.instance_sensors[0].angles_car_cosy_angle_sensor_dir2obj_deg,
                entity.instance_sensors[
                    0
                ].angles_car_cosy_angle_sensor_dir2eyes_dir_deg,
                entity.instance_sensors[0].angles_car_cosy_angle_sensor_dir2hip_dir_deg,
                entity.instance_sensors[0].sensor_occlusion_rate,
                entity.instance_sensors[0].sensor_occlusion_type,
                entity.instance_sensors[0].sensor_occlusion_total_pixels,
                entity.instance_sensors[0].sensor_occlusion_visible_pixels,
                entity.instance_sensors[
                    0
                ].sensor_occlusion_visible_skeleton_parts_from_joints,
                data_sample.meta_info.light_sources[0].instance_id,
                data_sample.meta_info.light_sources[0].world_angles_azimuth,
                data_sample.meta_info.light_sources[0].world_angles_elevation,
                data_sample.meta_info.light_sources[0].world_sky,
                data_sample.meta_info.light_sources[0].world_elevation,
                data_sample.meta_info.ego_sensors[0].instance_id,
                data_sample.meta_info.ego_sensors[0].angle_bev_north2fov_deg,
            ]

    # create data frame from dictionary
    data_columns = [
        "sample_name",
        "instance_id",
        "instance_pixels",
        "object_id",
        "occlusion",
        "occlusion_est",
        "rotation",
        "sensor",
        "size",
        "truncated",
        "velocity",
        "center",
        "class_id",
        "depth",
        "angle",
        "pos_cc",
        "pos_bev",
        "within_brake_dist_30kph",
        "within_brake_dist_50kph",
        "semantic_area",
        "eval_cat_a",
        "eval_cat_b",
        "prototype_asset_id",
        "prototype_mocap_asset",
        "prototype_ood_asset",
        "world_position_level_relative_diff_eye_lowest_bodypart",
        "world_semantic_area",
        "sensor_id",
        "plf_luminance_inst_dyn_range",
        "plf_contrast_rgb",
        "angle_sensor_dir2obj_deg",
        "angle_sensor_dir2eyes_dir_deg",
        "angle_sensor_dir2hip_dir_deg",
        "sensor_occlusion_rate",
        "sensor_occlusion_type",
        "sensor_occlusion_total_pixels",
        "sensor_occlusion_visible_pixels",
        "visible_skeleton_parts_from_joints",
        "light_source_instance_id",
        "light_source_angles_azimuth",
        "light_source_angles_elevation",
        "light_source_sky",
        "light_source_elevation",
        "ego_sensor_instance_id",
        "ego_sensor_angle_bev_north2fov_deg",
    ]
    return (
        pd.DataFrame.from_dict(data, orient="index", columns=data_columns),
        data_loader,
    )


def load_2dbb_predictions(
    backend: mbt_io.KIADatasetBackend,
    result_folder: str,
    config: mbt_io.KIADatasetConfig,
):
    """
    This function loads 2D bounding box detections into a data frame.
    """

    data = {}
    # create KIA reader with backend, result folder and configuration
    pred_reader = mbt_io.KIAReader(backend, result_folder, config)
    # iteratve over all samples and store them into dictionary
    for i in range(0, len(pred_reader)):
        predictions = pred_reader[i]
        for prediction in predictions.detections_2d:
            key = predictions.sample_name + "/" + str(prediction.instance_id)
            data[key] = [
                predictions.sample_name,
                prediction.instance_id,
                prediction.instance_pixels,
                prediction.object_id,
                prediction.occlusion,
                prediction.occlusion_estimate,
                prediction.rotation,
                prediction.sensor,
                prediction.size,
                prediction.truncated,
                prediction.velocity,
                prediction.center,
                prediction.class_id,
                prediction.depth,
                prediction.confidence,
            ]
    # create data frame from dictionary
    data_columns = [
        "sample_name",
        "instance_id",
        "instance_pixels",
        "object_id",
        "occlusion",
        "occlusion_est",
        "rotation",
        "sensor",
        "size",
        "truncated",
        "velocity",
        "center",
        "class_id",
        "depth",
        "confidence",
    ]
    return pd.DataFrame.from_dict(data, orient="index", columns=data_columns)
