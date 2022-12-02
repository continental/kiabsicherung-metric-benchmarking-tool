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
This files contains the configuration loader for the MBT tool as well as
dataclasses for each module.
"""

from dataclasses import dataclass, field
import json
from typing import List

import kia_mbt.kia_io.splits as splits


@dataclass
class IOConfig(object):
    """
    Dataclass for IO module configuration.

    Parameters
    ----------
        data_path : str
            Filepath of the root folder of the KIA dataset. Within in this path
            it is expected that there are the sequence folders.

        predictions_path : str
            The predctions path is the path to the folder, where the
            "predictions" folder is inside.

        results_folder : str
            The results folder that shall be used to load the predictions. The
            folder must be in <predictions_path>/predictions.

        backend : str
            The name of the backend that shall be used.

        sequences : List[str]
            A list of the KIA dataset sequences that shall be loaded.

        minio_endpoint : str
            When using the MinIO backend, this parameter specifies the endpoint.

        minio_bucket : str
            When using the MinIO backend, this parameter specifies the bucket.

        minio_use_proxy : str
            When using the MinIO backend, this paramter specifies whether a
            proxy shall be used or not. When enabled, it uses the proxy
            specified in the environment variable "http_proxy".
    """

    data_path: str = None
    predictions_path: str = None
    results_folder: str = None
    backend: str = "fs"
    sequences: List[str] = field(default_factory=list)
    minio_endpoint: str = None
    minio_bucket: str = None
    minio_use_proxy: bool = True

    def __post_init__(self):
        """
        Sets official KIA test split.

        When no sequences are given, the offical KIA test dataset split is set
        and will then be loaded. The offical test split contains the following
        tranches:
        - MV Tranche 4, 5 and 6
        - BIT-TS Tranche 3, 4 and 5
        Function is called after init is done.
        """

        # on default setting offical test split
        if not self.sequences:
            self.sequences = (
                splits.TEST_BIT_TRANCHE_3
                + splits.TEST_BIT_TRANCHE_4
                + splits.TEST_BIT_TRANCHE_5
                + splits.TEST_MV_TRANCHE_4
                + splits.TEST_MV_TRANCHE_5
                + splits.TEST_MV_TRANCHE_6
            )

    @classmethod
    def from_payload(cls, payload: dict):
        """
        Method to create IO configuration from payload dictionary.

        Parameters
        ----------
            payload : dict
                Dictionary with configuration from config file.

        Returns
        -------
        IO configuration parameters.
        """

        return cls(
            data_path=payload.get("data_path", None),
            predictions_path=payload.get("predictions_path", None),
            results_folder=payload.get("results_folder", None),
            backend=payload.get("backend", "fs"),
            sequences=payload.get("sequences", []),
            minio_endpoint=payload.get("minio_endpoint", None),
            minio_bucket=payload.get("minio_bucket", None),
            minio_use_proxy=payload.get("minio_use_proxy", True),
        )


@dataclass
class CorrelateConfig(object):
    """
    Dataclass for correlate module configuration.

    Parameters
    ----------
        iou_threshold : float
            Threshold used to determine if an annotation is correlated with a
            prediction.

        matching_type : str
            The type on how the matching is performed.

        clip_truncated_boxes : bool
            If True, clips boxes that are tower over the image.

        optional_arguments : dict
            Can contain additional configuration arguments for the correlation
            module. Please see documentation of the module.
    """

    iou_threshold: float = 0.1
    matching_type: str = "complete"
    clip_truncated_boxes: bool = True
    optional_arguments: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Sets default additional configuration arguments, when not specified.

        Is called after init.
        """

        if not self.optional_arguments:
            self.optional_arguments = {
                "confidence_col": "confidence",
                "annotation_bb_center_col": "center",
                "annotation_bb_size_col": "size",
                "detection_bb_center_col": "center",
                "detection_bb_size_col": "size",
            }

    @classmethod
    def from_payload(cls, payload: dict):
        """
        Method to create correlation configuration from payload dictionary.

        Parameters
        ----------
            payload : dict
                Dictionary with configuration from config file.

        Returns
        -------
        Correlation configuration parameters.
        """

        return cls(
            iou_threshold=payload.get("iou_threshold", 0.1),
            matching_type=payload.get("matching_type", "complete"),
            clip_truncated_boxes=payload.get("clip_truncated_boxes", True),
            optional_arguments=payload.get("optional_arguments", {}),
        )


@dataclass
class FilterConfig(object):
    """
    Dataclass for filter module configuration.

    Not that filters follow a three element dict or list form:

    ["column", "operator", "value"]

    Please see the filter module documentation for more details and look into
    the example configuration file.

    Parameters
    ----------
        annotation_filter : dict
            Filters that are applied to the annotation data.

        prediction_filter : dict
            Filters that are applied to the prediction data.

        matching_filter : dict
            Filters that are applied to the matching data.
    """

    annotation_filter: dict = field(default_factory=dict)
    prediction_filter: dict = field(default_factory=dict)
    matching_filter: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict):
        """
        Method to create filter configuration from payload dictionary.

        Parameters
        ----------
            payload : dict
                Dictionary with configuration from config file.

        Returns
        -------
        Filter configuration parameters.
        """

        return cls(
            annotation_filter=payload.get("annotation_filter", {}),
            prediction_filter=payload.get("prediction_filter", {}),
            matching_filter=payload.get("matching_filter", {}),
        )


@dataclass
class MetricConfig(object):
    """
    Dataclass for metric module configuration.

    Parameters
    ----------
        calculate : List[int]
            List of metric that shall be calculated. The list contains the
            metric identifiers. See also MetricProcessorFactory for a list of
            available metrics.

        parameters : dict
            Additonal parameters for the metric processors. This dictionary can
            contain configurations for multiple metric processors. See the
            example configuration file for an example.
    """

    calculate: List[int] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict):
        """
        Method to create metric configuration from payload dictionary.

        Parameters
        ----------
            payload : dict
                Dictionary with configuration from config file.

        Returns
        -------
        Metric configuration parameters.
        """

        return cls(
            calculate=payload.get("calculate", []),
            parameters=payload.get("parameters", {}),
        )

    def get_metric_parameters(self, metric_identifier: int) -> dict:
        """
        This method gets metric parameters for one metric by identifier.

        Parameters
        ----------
            metric_identifier : int
                Metric identifier.

        Returns
        -------
        Parameters of the metric.
        """

        for identifier, params in self.parameters.items():
            if int(identifier) == metric_identifier:
                return params
        return {}


@dataclass
class WriterConfig(object):
    """
    Dataclass for writer module configuration.

    Parameters
    ----------
        version_file : str
            Filepath of the version file.

        output_path : str
            Path where the calculated metrics shall be written to.
    """

    version_file: str = "version.json"
    output_path: str = ""

    @classmethod
    def from_payload(cls, payload: dict):
        """
        Method to create writer configuration from payload dictionary.

        Parameters
        ----------
            payload : dict
                Dictionary with configuration from config file.

        Returns
        -------
        Writer configuration parameters.
        """

        return cls(
            version_file=payload.get("version_file", "version.json"),
            output_path=payload.get("output_path", ""),
        )


class ConfigLoader(object):
    """
    Class for loading and handling configuration file.

    Parameters
    ----------
        config : dict
            Dictionary containing the loaded configuration file.
    """

    def __init__(self, config_filepath: str):
        """
        Opens configuration file on init and loads it.
        """

        with open(config_filepath, "r") as config_file:
            self.config = json.load(config_file)

    def get_io_config(self) -> IOConfig:
        """
        Get IO module configuration.

        Returns
        -------
        IO module configuration.
        """

        io_config = self.config["io"]
        return IOConfig.from_payload(io_config)

    def get_correlate_config(self) -> CorrelateConfig:
        """
        Get correlation module configuration.

        Returns
        -------
        Correlation module configuration.
        """

        correlate_config = self.config["correlate"]
        return CorrelateConfig.from_payload(correlate_config)

    def get_filter_config(self) -> FilterConfig:
        """
        Get filter module configuration.

        Returns
        -------
        Filter module configuration.
        """

        filter_config = self.config["filter"]
        return FilterConfig.from_payload(filter_config)

    def get_metric_config(self) -> MetricConfig:
        """
        Get metric module configuration.

        Returns
        -------
        Metric module configuration.
        """

        metric_config = self.config["metrics"]
        return MetricConfig.from_payload(metric_config)

    def get_writer_config(self) -> WriterConfig:
        """
        Get writer module configuration.

        Returns
        -------
        Writer module configuration.
        """

        writer_config = self.config["writer"]
        return WriterConfig.from_payload(writer_config)
