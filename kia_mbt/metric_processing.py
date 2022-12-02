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
This file contains a function to perform the metrics calculation.
"""

from typing import Tuple, List
import pandas as pd
import kia_mbt.config_loader as mbt_config
import kia_mbt.kia_metrics as mbt_metrics


def list_metrics() -> None:
    """
    This function lists all implemented metrics with name and ID.
    """

    factory = mbt_metrics.MetricProcessorFactory()
    processors = factory.get_all_processors()

    print("Available metrics:")
    for processor in processors:
        print("-", processor.name, "(ID:", str(processor.identifier) + ")")


def calc_metrics(
    annotation_data: pd.DataFrame,
    prediction_data: pd.DataFrame,
    matching: pd.DataFrame,
    config: mbt_config.MetricConfig,
) -> Tuple[List[Tuple[int, str, pd.DataFrame]], List[Tuple[int, str, pd.DataFrame]]]:
    """
    This function calculates available metrics.

    The function creates all available metric processors and executes them to
    get metric values. Thereby, only registered metric processors are executed.
    They can be registered in the MetricProcssorFactory.

    Parameters
    ----------
    annotation_data : pd.DataFrame
        Data frame with ground truth annotations.
    prediction_data : pd.DataFrame
        Data frame with predictions.
    matching : pd.DataFrame
        Data frame with the correlations between ground truth and predictions.
    config : MetricConfig
        Configuration for metric processing

    Returns
    -------
    Returns two data frames for sample-based and global-based metric results.

    """
    # create metric processors
    factory = mbt_metrics.MetricProcessorFactory()
    processors = []
    if config.calculate:
        processors = factory.get_processors(config.calculate)
    else:
        processors = factory.get_all_processors()

    # execute metric processors
    global_results = list()
    sample_results = list()
    for processor in processors:
        # get config parameters
        params = config.get_metric_parameters(processor.identifier)
        # compute global metric results
        result = processor.calc_global(
            annotation_data=annotation_data,
            prediction_data=prediction_data,
            matching=matching,
            **params
        )
        global_results.append(
            (
                processor.identifier,
                processor.name,
                result,
            )
        )
        # compute sample-based metric results when possible
        if processor.calculate_per_sample:
            result = processor.calc_per_sample(
                annotation_data=annotation_data,
                prediction_data=prediction_data,
                matching=matching,
                **params
            )
            sample_results.append(
                (
                    processor.identifier,
                    processor.name,
                    result,
                )
            )

    return global_results, sample_results
