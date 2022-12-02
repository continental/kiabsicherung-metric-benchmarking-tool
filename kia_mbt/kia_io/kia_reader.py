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
This file contains the reader for the KIA output format

"""

import os
from typing import List
import numpy as np
from kia_mbt.kia_io import (
    KIADatasetBackend,
    KIADatasetConfig,
    KIADetection2D,
    KIAPredictionContainer,
)
import kia_mbt.kia_io.constants as constant


class KIAReader(object):
    """
    Reader for the KIA data in output format

    The KIA reader can read result data which follow the specification of
    the defined result formates. Basically, the reader can read predictions
    from a ML model, e.g. 2d bounding box prediction, as specified in the
    project. The folder structure shall be the following:

    <ROOT_PATH>/predictions/<RESULT_FOLDER>

    ROOT_PATH is the root folder where results are stored and RESULT_FOLDER is
    the name of a specific result, e.g. Opel-SSD-r3-v2.

    Plase note that this reader is implemented based on the following
    specification: https://confluence.vdali.de/x/9wdyAw

    Attributes
    ----------
        backend : KIADatasetBackend
            Backend to perform the read operations from a storage
        sample_tokens : List[str]
            A list with all sample tokens that can be accessed
        result_folder : str
            Relative path to the concrete result folder
    """

    def __init__(
        self, backend: KIADatasetBackend, result_folder: str, config: KIADatasetConfig
    ) -> None:
        """
        Setup of the KIA reader

        Parameters
        ----------
            backend : KIABackend
                Backend to perform the write operation to a storage.
            result_folder : str
                Result folder, e.g. "Opel-SSD-r3-v1"
            config : KIADatasetConfig
                Configuration of which data to read from the dataset.
            use_eval _ bool
                If true, the evaluations folder is used otherwise the
                predictions folder is used.
        """

        self.result_folder = os.path.join(constant.FOLDER_PRED, result_folder).replace(
            os.sep, "/"
        )

        self.backend = backend
        self.sample_tokens = self._load_sample_tokens(config)

    def _load_sample_tokens(self, config: KIADatasetConfig) -> List[str]:
        """
        Load sample tokens with filtering

        The method loads all object name from result folder and filters the
        object names in dependency of the dataset configuration. Afterwards
        object names are encoded into unique sample tokens.

        Parameters
        ----------
            config : KIADatasetConfig
                Dataset configuration

        Returns
        -------
        List of sample tokes
        """

        # load all object names from result folder
        object_names = self.backend.get_object_names(self.result_folder)
        if not object_names:
            raise IOError("Result folder {} does not exist".format(self.result_folder))

        sample_tokens = []
        # filter object names
        if config.sequences:
            # filter by dataset sequences
            sample_tokens = self._filter_objects_by_sequences(
                object_names, config.sequences
            )
        elif config.sequence_names:
            sample_tokens = self._filter_objects_by_sequence_names(
                object_names, config.sequence_names
            )
        else:
            # filter by dataset configuration
            if config.tranches:
                sample_tokens = self._filter_objects_by_config(
                    object_names, config.tranches, config.company, config.dataset_split
                )
            elif config.company or config.dataset_split:
                sample_tokens = self._filter_objects_by_config(
                    object_names,
                    constant.KIA_DATASET_TRANCHES,
                    config.company,
                    config.dataset_split,
                )
            else:
                # Load all available data
                sample_tokens = self._objects_to_sample_tokens(object_names)

        return sample_tokens

    def _filter_objects_by_config(
        self, objects: List[str], tranches: List[int], company: str, dataset_split: str
    ) -> List[str]:
        """
        Filters a given object list by a configuration.

        The configuration can contain different data tranches, data producing
        companies or different datasplits. In addition, objects names are
        translated into sample tokens.

        Parameters
        ----------
            objects : List[str]
                List of object names
            tranches : List[int]
                List of tranche numbers
            company : str
                Company name
            dataset_split : str
                Name of the dataset split

        Returns
        -------
        Filtered list of sample tokens.
        """

        sequences = []
        for tranche in tranches:
            split = constant.KIA_DATASET_SPLITS[tranche]
            companies = ["bit", "mv"]
            if company:
                companies = [company]
            for c in companies:
                if dataset_split:
                    sequences = sequences + split[c][dataset_split]
                else:
                    sequences = sequences + split[c]["train"]
                    sequences = sequences + split[c]["val"]
                    sequences = sequences + split[c]["test"]
        # Extract sequence hash from each sequence string
        sequences_hash = []
        for seq in sequences:
            if "-" in seq:
                # BIT-TS sequence name
                sequences_hash.append(seq.split("-")[1])
            else:
                # MV sequence name
                sequences_hash.append(seq.split("_")[-1])
        sequences_hash = "\t".join(sequences_hash)
        objects = [
            obj for obj in objects if self._get_sample_hash(obj) in sequences_hash
        ]
        return self._objects_to_sample_tokens(objects)

    def _filter_objects_by_sequences(
        self, objects: List[str], sequences: List[int]
    ) -> List[str]:
        """
        Filters given object list by sequence numbers

        Parameters
        ----------
            objects : List[str]
                List of object names
            sequences : List[int]
                List of sequences numbers
        """

        filtered_objects = []
        for obj in objects:
            if int(obj.split("-")[-3]) in sequences:
                filtered_objects.append(obj)
        return self._objects_to_sample_tokens(filtered_objects)

    def _filter_objects_by_sequence_names(
        self, objects: List[str], sequences: List[str]
    ) -> List[str]:
        filtered_objects = []
        for obj in objects:
            object_hash = obj.split("-")[-2]
            if any(object_hash in ele for ele in sequences):
                filtered_objects.append(obj)
        return self._objects_to_sample_tokens(filtered_objects)

    def _get_sample_hash(self, object_name: str) -> str:
        """
        Get sample hash from object name

        Parameters
        ----------
            object_name : str
                Object name

        Returns
        -------
        Hash of the object
        """

        sample_name = object_name.split("/")[-1]
        return sample_name.split("-")[3]

    def _objects_to_sample_tokens(self, objects: List[str]) -> List[str]:
        """
        Converts object names to sample tokens.

        Parameters
        ----------
            objects : List[str]
                List of object names

        Returns
        -------
        List of sample tokens
        """

        sample_tokens = []
        for obj in objects:
            tokens = obj.split("/")
            # TODO: check if hash is mv or bit
            obj_hash = self._get_sample_hash(obj)
            company = self._company_from_seq_hash(obj_hash)
            if company == "mv":
                sample_tokens.append(("mv/" + tokens[-1].replace(".json", "")))
            elif company == "bit":
                sample_tokens.append(("bit/" + tokens[-1].replace(".json", "")))
            else:
                print("Unknown sequence: {}".format(tokens[-1].replace(".json", "")))
        return sample_tokens

    def _company_from_seq_hash(self, sequence_hash: str) -> str:
        """
        Get company name from sequence hash.

        Parameters
        ----------
            sequence_hash : str
                Sequence hash

        Returns
        -------
        Name of the company.
        """

        for tranche in constant.KIA_DATASET_SPLITS.values():
            for company, splits in tranche.items():
                for split in splits.values():
                    if sequence_hash in "\t".join(split):
                        return company
        return ""

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

    def _get_sensor(self, sample_token: str) -> str:
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

    def _get_split(self, sample_token: str) -> str:
        """
        Get the split name

        Split name can be either train, val or test. The split is determined by
        the official defined data split in KI Absicherung. If the sample token
        cannot be mapped to a dataset split, a LookupError exception will be
        raised.

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        Split name
        """

        sequence_hash = self._get_sample_hash(sample_token)
        for tranche in constant.KIA_DATASET_SPLITS.values():
            for splits in tranche.values():
                for split, sequences in splits.items():
                    if sequence_hash in "\t".join(sequences):
                        return split
        raise LookupError(
            "Sample token {} could not be maped to a dataset split.".format(
                sample_token
            )
        )

    def get_samples(self) -> List[str]:
        """
        Get all samples

        Returns
        -------
        List of all sample tokens.
        """

        return self.sample_tokens

    def read_predictions_2d(self, sample_token: str) -> List[KIADetection2D]:
        """
        Read the 2D bounding box predictions of a sample

        Parameters
        ----------
            sample_token : str
                Name of a sample token.

        Returns
        -------
        2D bounding box predictions of a sample.
        """

        object_name = "{}/2d-bounding-box_json/{}/{}.json".format(
            self.result_folder,
            self._get_split(sample_token),
            self._get_frame(sample_token),
        )
        data = self.backend.get_json_object(object_name)

        detections_2d = []
        for obj_id, values in data.items():
            if not "__" in obj_id:
                if "objectness_score" in values:
                    confidence = values["objectness_score"]
                else:
                    confidence = values["confidence"] if "confidence" in values else 1.0

                # Fill all non existing fields with default values.
                occlusion = values["occlusion"] if "occlusion" in values else -1
                truncated = values["truncated"] if "truncated" in values else False
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
                    center = (
                        values["center"] if "center" in values else [np.nan, np.nan]
                    )
                    size = values["size"] if "size" in values else [np.nan, np.nan]
                    velocity = (
                        values["velocity"] if "velocity" in values else [np.nan, np.nan]
                    )
                instance_id = (
                    values["instance_id"] if "instance_id" in values else int(obj_id)
                )
                object_id = (
                    values["object_id"] if "object_id" in values else int(obj_id)
                )
                depth = values["depth"] if "depth" in values else -1.0
                instance_pixels = (
                    values["instance_pixels"] if "instance_pixels" in values else -1
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
                    sensor=self._get_sensor(sample_token),
                    center=np.array(center),
                    size=np.array(size),
                    rotation=0,
                    confidence=confidence,
                    occlusion=occlusion,
                    occlusion_estimate=-1.0,
                    velocity=np.array(velocity),
                    truncated=truncated,
                    instance_id=instance_id,
                    object_id=object_id,
                    depth=depth,
                    instance_pixels=instance_pixels,
                )
                detections_2d.append(detection)

        return detections_2d

    def __len__(self) -> int:
        """
        Get number of sample tokens
        """

        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> KIAPredictionContainer:
        """
        Get sample data

        Parameters
        ----------
            idx : int
                Sample index

        Returns
        -------
        Data of sample
        """

        data = KIAPredictionContainer()
        sample_token = self.sample_tokens[idx]
        data.sample_name = sample_token

        data.detections_2d = self.read_predictions_2d(sample_token)

        return data
