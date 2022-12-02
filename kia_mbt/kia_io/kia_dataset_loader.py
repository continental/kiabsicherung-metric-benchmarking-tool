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
This file contains the frontend KIA dataset loader

"""

from typing import List, Mapping, Any
import collections.abc
from PIL import Image
import numpy as np
from kia_mbt.kia_io.backend import KIADatasetBackend
from kia_mbt.kia_io.types import (
    KIAAdditionalMetaInformation,
    KIADataContainer,
    KIADatasetConfig,
    KIADetection2D,
    KIAEgoSensorInformation,
    KIAEntityInformaton,
    KIAInstanceSensorInformation,
    KIALightSourceInformation,
)
import kia_mbt.kia_io.constants as constant


class KIADatasetLoader(object):
    """
    This class implements the frontend KIA dataset loader

    The KIA dataset loader requires a backend to access the KIA data. There are
    currently the following backends:
    - MinIO backend (class KIADatasetMinIOBackend)
    Apart from the backend also a configuration is required. Please see the
    KIADatasetConfig type for more details.

    Attributes
    ----------
        backend : KIADatasetBackend
            Backend object

        config : KIADatasetConfig
            Configuration object

        all_sample_tokens: List[str]
            A list of all sample tokens (object names) of the KIA dataset
    """

    def __init__(self, backend: KIADatasetBackend, config: KIADatasetConfig) -> None:
        """
        Constructs the KIA dataset loader with the given backend and configuration

        When the KIA dataset loader is constructed, it will try to access the
        dataset via the backend and loads all samples (object names) to create
        a list containing all data samples for faster data access. This might
        take some while.

        Parameters
        ----------
            backend : KIADatasetBackend
                Backend object
            config : KIADatasetConfig
                Configuration object
        """

        self.backend = backend
        self.config = config
        self.all_sample_tokens = []

        # loading sample tokens
        if not config.sequences and not config.sequence_names:
            # load sample tokens by other configuration (tranches, split and company)
            self.all_sample_tokens = self._load_samples_by_config(config)
        elif not config.sequences:
            # load sample tokens by sequences names
            self.all_sample_tokens = self._load_all_sample_tokens_by_seq_names(
                config.sequence_names
            )
        else:
            # load sample tokens by sequence numbers and company filter
            self.all_sample_tokens = self._load_all_sample_tokens(
                config.sequences, config.company
            )

    def _load_samples_by_config(self, config: KIADatasetConfig) -> List[str]:
        """
        Load sample tokens by configuration

        This method loads data samples by the configuration regarding tranches,
        data producing company and dataset split. If the configuration is empty,
        all sample tokens will be loaded.

        Parameters
        ----------
            config : KIADatasetConfig
                Configuration

        Returns
        -------
        List with sample tokens.
        """

        if config.tranches:
            # load sample tokens by selected tranches with company and dataset split filter
            sequences = self._get_sequence_names_by_config(
                config.tranches, config.company, config.dataset_split
            )
            sample_tokens = self._load_all_sample_tokens_by_seq_names(sequences)
        elif config.company or self.config.dataset_split:
            # load sample tokens from all tranches, filtered by company and/or dataset split
            sequences = self._get_sequence_names_by_config(
                constant.KIA_DATASET_TRANCHES, config.company, config.dataset_split
            )
            sample_tokens = self._load_all_sample_tokens_by_seq_names(sequences)
        else:
            # Empty configuration, loading all sample tokens
            sample_tokens = self._load_all_sample_tokens([], config.company)
        return sample_tokens

    def _get_sequence_names_by_config(
        self, tranches: List[int], company: str, dataset_split: str
    ) -> List[str]:
        """
        Get list of sequence names by configuration.

        This methods returns a list of sequence names, that is defined by the
        tranches, company and dataset split configuration.

        Parameters
        ----------
            tranches : List[int]
                List with tranche numbers

            company : str
                Filter for company name, e.g. bit

            dataset_split : str
                Name of the dataset split, e.g. train.

        Returns
        -------
        Returns a list of sequence names according to the configuration.
        """

        sequences = []
        for tranche in tranches:
            split = constant.KIA_DATASET_SPLITS[tranche]
            companies = [company] if company else ["bit", "mv"]
            for comp in companies:
                if dataset_split:
                    sequences = sequences + split[comp][dataset_split]
                else:
                    sequences = sequences + split[comp]["train"]
                    sequences = sequences + split[comp]["val"]
                    sequences = sequences + split[comp]["test"]
        return sequences

    def _load_all_sample_tokens(self, sequences: List[int], company: str) -> List[str]:
        """
        Get all samples tokens from backend

        Private method which loads all sample tokens from the backend and stores
        them into a list. Note that sequences and company are used to filter the
        data. Please also see the data class KIADatasetConfig.

        Parameters
        ----------
            sequence : List[int]
                A list of sequence numbers, which shall be loaded
            company: str
                Filter to only load data from a certain data production company

        Returns
        -------
        List with sample tokens.
        """

        frames = [
            f for f in self.backend.get_object_names() if "sensor/camera/left/png/" in f
        ]
        sample_tokens = []
        for f in frames:
            tokens = f.split("/")
            # filter if by sequences if list is not empty
            if sequences:
                sequence = tokens[0].split("_")[3].split("-")[0]
                if int(sequence) not in sequences:
                    continue
            # filter by company
            if tokens[0].startswith("mv_"):
                if not company or company == "mv":
                    sample_tokens.append(("mv/" + tokens[-1].replace(".png", "")))
            elif tokens[0].startswith("bit_"):
                if not company or company == "bit":
                    sample_tokens.append(("bit/" + tokens[-1].replace(".png", "")))
            else:
                print(
                    "Unknown sequence prefix: {}".format(sample_tokens[0].split("_")[0])
                )
        return sample_tokens

    def _load_all_sample_tokens_by_seq_names(self, sequences: List[str]) -> List[str]:
        """
        Get all sample tokens from backend by sequence names

        Private method which loads all samples from the given sequence names.

        Parameters
        ----------
        sequences : List[str]
            List with sequence names.

        Returns
        -------
        List with sample tokens.
        """

        frames = []
        for seq in sequences:
            frames = frames + [
                f
                for f in self.backend.get_object_names(seq)
                if "sensor/camera/left/png/" in f
            ]
        sample_tokens = []
        for f in frames:
            tokens = f.split("/")
            company = tokens[0].split("_")[0]
            sample_tokens.append((company + "/" + tokens[-1].replace(".png", "")))
        return sample_tokens

    def get_sample_tokens(self) -> List[str]:
        """
        Get all sample tokens.

        Returns
        -------
        Returns a list of strings containing all (filtered) sample tokens.
        """

        return self.all_sample_tokens

    def __len__(self) -> int:
        """
        Get number of loaded sample tokens.

        Returns
        -------
        Number of loaded (and filtered) sample tokens.
        """

        return len(self.all_sample_tokens)

    def _get_sequence(self, sample_token: str) -> str:
        """
        Get the sequence name of a sample token.

        A sample token has the following structure:
        {CompanyName}/{CamType}-camera{CamID}-{SequenceID}-{SequenceUUID}-{FrameID}
        The sequence name for BIT-TS is then:
        {CompanyName}_results_sequence_{SequenceID}-{SequenceUUID}
        For MV it is:
        {CompanyName}_results_sequence_{SequenceID}_{SequenceUUID}

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Sequence name.
        """

        # check if sample token is BIT-TS or MV
        delimeter = "-"
        if sample_token.split("/")[0] == "mv":
            delimeter = "_"

        return (
            sample_token.split("/")[0]
            + "_results_sequence_"
            + sample_token.split("-")[2]
            + delimeter
            + sample_token.split("-")[3]
        )

    def _get_frame(self, sample_token: str) -> str:
        """
        Get the frame or file name of a sample token.

        A sample token has the following structure:
        {CompanyName}/{CamType}-camera{CamID}-{SequenceID}-{SequenceUUID}-{FrameID}
        The frame or file name is then:
        {CamType}-camera{CamID}-{SequenceID}-{SequenceUUID}-{FrameID}

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Frame or file name.
        """

        return sample_token.split("/")[1]

    def _get_world(self, sample_token: str) -> str:
        """
        Get the world file name.

        A sample token has the following structure:
        {CompanyName}/{CamType}-camera{CamID}-{SequenceID}-{SequenceUUID}-{FrameID}
        The world file name is then:
        world-{SequenceID}-{SequenceUUID}-{FrameID}

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Frame or file name.
        """
        frame_file_name = sample_token.split("/")[1].split("-")
        world_file_name = (
            "world-"
            + frame_file_name[2]
            + "-"
            + frame_file_name[3]
            + "-"
            + frame_file_name[4]
        )
        return world_file_name

    @staticmethod
    def _get_sensor(sample_token: str) -> str:
        """
        Get the sensor name of a sample token.

        A sample token has the following structure:
        {CompanyName}/{CamType}-camera{CamID}-{SequenceID}-{SequenceUUID}-{FrameID}
        The sensor name is then:
        {CamType}-camera{CamID}

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Sensor name.
        """

        return (
            sample_token.split("-")[0].split("/")[-1] + "-" + sample_token.split("-")[1]
        )

    def get_image_exr(self, sample_token: str) -> Image:
        """
        Get the EXR (raw) image of an sample token.

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Return the EXR image as PIL image.
        """

        return self._get_image(sample_token, "exr")

    def get_image_png(self, sample_token: str) -> Image.Image:
        """
        Get the PNG (compressed) image of an sample token.

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Return the PNG image as PIL image.
        """

        return self._get_image(sample_token, "png")

    def _get_image(self, sample_token: str, filetype: str) -> Image.Image:
        """
        Get image object from backend with specified filetype.

        Parameters
        ----------
            sample_token : str
                Name of a sample token.
            filetype : str
                Name of the file extension, e.g. png or exr.

        Returns
        -------
        Image object as PIL image with given type.
        """

        filename = "{0}/sensor/camera/left/{2}/{1}.{2}".format(
            self._get_sequence(sample_token), self._get_frame(sample_token), filetype
        )
        return self.backend.get_image_object(filename)

    def get_semantic_segmentation(self, sample_token: str) -> Image.Image:
        """
        Get the semantic group segmentation image

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Semantic group segmentation image as PIL image.
        """

        filename = "{}/ground-truth/semantic-group-segmentation_png/{}.png".format(
            self._get_sequence(sample_token), self._get_frame(sample_token)
        )
        return self.backend.get_image_object(filename)

    def get_instance_segmentation(self, sample_token: str) -> Image.Image:
        """
        Get the semantic instance segmentation image

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Semantic instance segmentation image as PIL image.
        """

        fname = "{}/ground-truth/semantic-instance-segmentation_png/{}.png".format(
            self._get_sequence(sample_token), self._get_frame(sample_token)
        )
        if self.backend.exists_object_name(fname):
            # E1.2.3 official format
            img_instance = self.backend.get_image_object(fname)
        else:
            # Legacy format from early releases
            fname = "{}/ground-truth/semantic-instance-segmentation_exr/{}.exr".format(
                self._get_sequence(sample_token), self._get_frame(sample_token)
            )
            img_instance = self.backend.get_image_object(fname)
            # TODO: img_instance = img_instance[:,:,2].astype('uint16') for OpenCV but required for PIL Image
            raise NotImplementedError(
                "Loading of instance segmentation for EXRs not correctly implemented yet."
            )
        return img_instance

    def get_body_part_segmentation(self, sample_token: str) -> Image.Image:
        """
        Get the body part segmentation image

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Body part segmentation image.
        """

        fname = "{}/ground-truth/body-part-segmentation_png/{}.png".format(
            self._get_sequence(sample_token), self._get_frame(sample_token)
        )
        if self.backend.exists_object_name(fname):
            body_part_seg = self.backend.get_image_object(fname)
        else:
            raise FileNotFoundError(
                (
                    "Body part segmentation is not available for"
                    "\ntoken {}\nwith object name {}."
                ).format(sample_token, fname)
            )
        return body_part_seg

    def get_detections_2d(self, sample_token: str) -> List[KIADetection2D]:
        """
        Get the 2D bounding box annotations of a sample

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        2D bounding box annotations of a sample.
        """

        # Load 2D bounding box annotation from available files
        # order: enriched, fixed, default
        object_name = "{}/ground-truth/{}/{}.json".format(
            self._get_sequence(sample_token),
            constant.FOLDER_2DBB_ENRICHED,
            self._get_frame(sample_token),
        )
        if not self.backend.exists_object_name(object_name):
            # if enriched folder is not available, try fixed folder
            object_name = "{}/ground-truth/{}/{}.json".format(
                self._get_sequence(sample_token),
                constant.FOLDER_2DBB_FIXED,
                self._get_frame(sample_token),
            )
            if not self.backend.exists_object_name(object_name):
                # if fixed folder is not availabe use default
                object_name = "{}/ground-truth/{}/{}.json".format(
                    self._get_sequence(sample_token),
                    constant.FOLDER_2DBB,
                    self._get_frame(sample_token),
                )

        # get 2d bounding box annotations as dictionary from JSON
        data: Mapping[str, Mapping[str, Any]] = self.backend.get_json_object(
            object_name
        )
        assert isinstance(data, collections.abc.Mapping)

        detections_2d = []

        # iterate over all instances
        for key in data.keys():
            if key == "info":  # ignore info part in enriched data
                continue
            detection = self.json_entry_to_detection(data[key], sample_token, key)
            detections_2d.append(detection)

        return detections_2d

    @classmethod
    def json_entry_to_detection(
        cls, values: Mapping[str, Any], sample_token: str, instance_key: str
    ) -> KIADetection2D:
        """
        Stores the JSON entries into the detection data class.

        Parameters
        ----------
            values
                The values of the JSON dictionary
            sample_token
                Name of a sample token
            instance_key
                Identifier of the instance
        """

        if "center" not in values:  # Official E1.2.3 (V3.0 mode)
            center = (
                [values["c_x"], values["c_y"]]
                if "c_x" in values and "c_y" in values
                else [np.nan, np.nan]
            )
            size = (
                [values["w"], values["h"]]
                if "w" in values and "h" in values
                else [np.nan, np.nan]
            )
            velocity = (
                [values["v_x"], values["v_y"]]
                if "v_x" in values and "v_y" in values
                else [np.nan, np.nan]
            )
        else:  # DFKI KIASampleWriter format
            center = values.get("center", [np.nan, np.nan])
            size = values.get("size", [np.nan, np.nan])
            velocity = values.get("velocity", [np.nan, np.nan])

        pos_cc = (
            [values["pos_cc_x"], values["pos_cc_y"]]
            if "pos_cc_x" in values and "pos_cc_y" in values
            else [np.nan, np.nan]
        )
        pos_bev = (
            [values["pos_bev_col"], values["pos_bev_row"]]
            if "pos_bev_col" in values and "pos_bev_row" in values
            else [np.nan, np.nan]
        )

        class_id = "unknown"
        if "class_id" in values:
            class_id = values["class_id"]
        elif "class" in values:
            class_id = values["class"]
        elif "category" in values:
            class_id = values["category"]

        detection = KIADetection2D(
            class_id=class_id,
            sensor=cls._get_sensor(sample_token),
            center=np.array(center),
            size=np.array(size),
            rotation=0,
            confidence=float(
                values.get("confidence", 1.0)
            ),  # Confidence only exists for predictions => default to 1,
            occlusion=float(values.get("occlusion", -1)),
            occlusion_estimate=float(values.get("occlusion_est", -1.0)),
            velocity=np.array(velocity),
            truncated=bool(values.get("truncated", False)),
            instance_id=int(values.get("instance_id", instance_key)),
            object_id=int(values.get("object_id", instance_key)),
            depth=float(values.get("depth", -1.0)),
            instance_pixels=int(values.get("instance_pixels", -1)),
            angle=float(values.get("angle", 0)),
            pos_cc=np.array(pos_cc),
            pos_bev=np.array(pos_bev),
            within_brake_dist_30kph=values.get("within_brake_dist_30kph", None),
            within_brake_dist_50kph=values.get("within_brake_dist_50kph", None),
            semantic_area=str(values.get("semantic_area", "")),
            eval_cat_a=values.get("eval_catA", None),
            eval_cat_b=values.get("eval_catB", None),
        )

        return detection

    def get_additional_meta_info(
        self, sample_token: str
    ) -> KIAAdditionalMetaInformation:
        """
        Loads additional meta information.

        This function loads selected attributes from the
        general-globally-per-frame-analysis-enriched_json folder.

        Parameters
        ----------
            sample_token : str
                Name of a sample token

        Returns
        -------
        Additional meta information.
        """

        # load object which contain additional meta info
        object_name = "{}/ground-truth/{}/{}.json".format(
            self._get_sequence(sample_token),
            constant.FOLDER_META_INFO,
            self._get_world(sample_token),
        )
        if not self.backend.exists_object_name(object_name):
            # if additional meta info file does not exist, return empty list
            return []

        data = self.backend.get_json_object(object_name)

        meta_info = KIAAdditionalMetaInformation()
        for key in data["base_context"].keys():
            if key == "entities":
                meta_info.entities = self.get_meta_info_entities(
                    data["base_context"][key]
                )
            elif key == "light_source":
                meta_info.light_sources = self.get_meta_info_light_sources(
                    data["base_context"][key]
                )
            elif key == "ego_sensors":
                meta_info.ego_sensors = self.get_meta_info_ego_sensors(
                    data["base_context"][key]
                )

        return meta_info

    def get_meta_info_entities(self, data_entities) -> List[KIAEntityInformaton]:
        """
        Store entities meta information into data structure.

        Note that only entities which are of type pedestrian will be stored.

        Parameters
        ----------
            data_entities
                Dictionary with all meta information from the entities key

        Returns
        -------
        List of filtered entity meta information.
        """

        entities = []
        for instance_id, data_entity in data_entities.items():
            # ignore all entities that are not from type pedestrian
            if "type" in data_entity:
                if not data_entity["type"] == "pedestrian":
                    continue
            else:
                continue

            # get instances sensors information of the entity
            sensors = []
            for key in data_entity.keys():
                if "car" in key or "arb" in key:
                    data_sensor = data_entity[key]

                    sensor_occlusion_rate = np.nan
                    if "rate_corrected" in data_sensor["sensor_occlusion"]:
                        sensor_occlusion_rate = data_sensor["sensor_occlusion"][
                            "rate_corrected"
                        ]
                    elif "rate" in data_sensor["sensor_occlusion"]:
                        sensor_occlusion_rate = data_sensor["sensor_occlusion"]["rate"]

                    sensor = KIAInstanceSensorInformation(
                        sensor_id=key,
                        angles_car_cosy_angle_sensor_dir2obj_deg=data_sensor[
                            "angles_car_cosy"
                        ].get("angle_sensor_dir2obj_deg", np.nan),
                        angles_car_cosy_angle_sensor_dir2hip_dir_deg=data_sensor[
                            "angles_car_cosy"
                        ]["angle_sensor_dir2hip_dir_deg"],
                        angles_car_cosy_angle_sensor_dir2eyes_dir_deg=data_sensor[
                            "angles_car_cosy"
                        ]["angle_sensor_dir2eyes_dir_deg"],
                        metainfo_plf_contrast_rgb=float(
                            data_sensor["metainfo_plf"].get("contrast_rgb", np.nan)
                        ),
                        metainfo_plf_luminance_inst_dyn_range=float(
                            data_sensor["metainfo_plf"].get(
                                "luminance_inst_dyn_range", np.nan
                            )
                        ),
                        sensor_occlusion_rate=sensor_occlusion_rate,
                        sensor_occlusion_type=str(
                            data_sensor["sensor_occlusion"].get("type", None)
                        ),
                        sensor_occlusion_total_pixels=int(
                            data_sensor["sensor_occlusion"].get("total_pixels", -1)
                        ),
                        sensor_occlusion_visible_pixels=int(
                            data_sensor["sensor_occlusion"].get("visible_pixels", -1)
                        ),
                        sensor_occlusion_visible_skeleton_parts_from_joints=data_sensor[
                            "sensor_occlusion"
                        ]["visible_skeleton_parts_from_joints"],
                    )

                    sensors.append(sensor)

            # store attributes into entity data structure
            entity = KIAEntityInformaton(
                instance_id=int(instance_id),
                prototype_asset_id=data_entity["prototype"]["asset_id"],
                prototype_ood_asset=data_entity["prototype"].get("ood_asset", None),
                prototype_mocap_asset=data_entity["prototype"].get("mocap_asset", None),
                world_semantic_area=data_entity["world"]["position"].get(
                    "semantic_area", None
                ),
                world_position_level_relative_diff_eye_lowest_bodypart=data_entity[
                    "world"
                ]["position"]["level_relative"]["diff_eye_lowest_bodypart"],
                instance_sensors=sensors,
            )

            entities.append(entity)

        return entities

    def get_meta_info_light_sources(
        self, data_light_sources
    ) -> List[KIALightSourceInformation]:
        """
        Store light sources meta information into data structure.

        Parameters
        ----------
            data_light_sources
                Dictionary with all attributes of the light sources.

        Returns
        -------
        List of filtered light source meta information.
        """

        light_sources = []

        for instance_id, data_light_source in data_light_sources.items():
            light_source = KIALightSourceInformation(
                instance_id=instance_id,
                world_angles_azimuth=data_light_source["world"]["angles"]["azimuth"],
                world_angles_elevation=data_light_source["world"]["angles"][
                    "elevation"
                ],
                world_sky=data_light_source["world"]["sky"],
                world_elevation=data_light_source["world"].get("elevation", None),
            )
            light_sources.append(light_source)

        return light_sources

    def get_meta_info_ego_sensors(
        self, data_ego_sensors
    ) -> List[KIAEgoSensorInformation]:
        """
        Store ega sensors meta information into data structure.

        Parameters
        ----------
            data_ego_sensors
                Dictionary of all ego sensors attributes.

        Returns
        -------
        List of filtered ego sensor meta information.
        """

        ego_sensors = []

        for instance_id, data_ego_sensor in data_ego_sensors.items():

            # ignore world information
            if instance_id == "world":
                continue

            ego_sensor = KIAEgoSensorInformation(
                instance_id=instance_id,
                angle_bev_north2fov_deg=data_ego_sensor["angles_bev"][
                    "angle_bev_north2fov_deg"
                ],
            )
            ego_sensors.append(ego_sensor)

        return ego_sensors

    def __getitem__(self, idx: int) -> KIADataContainer:
        """
        Get sample from backend.

        This method enables the loading of all samples, by accessing the
        KIADatasetLoader object with [idx]. Note that only data object are
        loaded, which are enabled through the configuration of the loader.

        Parameters
        ----------
            idx : int
                List identifier for accessing a sample

        Returns
        -------
        Data container storing the loaded objects of a sample.
        """

        data = KIADataContainer()
        sample_token = self.all_sample_tokens[idx]
        data.sample_name = sample_token

        if self.config.get_image_png:
            data.image_png = self.get_image_png(sample_token)

        if self.config.get_image_exr:
            data.image_exr = self.get_image_exr(sample_token)

        if self.config.get_grp_seg:
            data.grp_seg = self.get_semantic_segmentation(sample_token)

        if self.config.get_inst_seg:
            data.inst_seg = self.get_instance_segmentation(sample_token)

        if self.config.get_body_part:
            data.body_part = self.get_body_part_segmentation(sample_token)

        if self.config.get_detections_2d:
            data.detections_2d = self.get_detections_2d(sample_token)

        if self.config.get_meta_info:
            data.meta_info = self.get_additional_meta_info(sample_token)

        return data
