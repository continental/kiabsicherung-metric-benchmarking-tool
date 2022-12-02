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
This file contains the interface class for all metric processors.

"""

from abc import ABC
import pandas as pd
import numpy as np


class MetricProcessor(ABC):
    """
    Base class for all metric processors
    """

    def __init__(self, identifier: int, name: str, calculate_per_sample: bool) -> None:
        """
        Setup the identifier and name of the metric.

        Parameters
        ----------
            identifier : int
                Identifier of the metric according to assigned metric ID in the
                KI Absicherung project.

            name : str
                Human readable name of the metric.

            calculate_per_sample : bool
                Flag indicating if this metric processor computes also metrics
                per sample.
        """

        self.identifier = identifier
        self.name = name
        self.calculate_per_sample = calculate_per_sample

    def calc(
        self,
        annotation_data: pd.DataFrame,
        prediction_data: pd.DataFrame,
        matching: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Prototype to calculate the data metric for a given input matching data frame.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

            kwargs
                Configuration parameters as dictionary for the metric
                calculation.

        Returns
        -------
        Data frame containing the calculated metric(s).

        """

        raise NotImplementedError

    def calc_global(
        self,
        annotation_data: pd.DataFrame,
        prediction_data: pd.DataFrame,
        matching: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate the data metric for the entire input matching data frame with
        applied iou and confidence thresholds.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

            kwargs
                Configuration parameters as dictionary for the metric
                calculation.

        Returns
        -------
        Data frame containing the calculated metric(s).

        """

        global_metric = self.calc(
            annotation_data=annotation_data,
            prediction_data=prediction_data,
            matching=matching,
            **kwargs
        )
        return global_metric

    def calc_per_sample(
        self,
        annotation_data: pd.DataFrame,
        prediction_data: pd.DataFrame,
        matching: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate the data metric over the per sample.

        Provides basic implementation for computing the metric
        by iterating over samples and applying the metric calulation
        to conditioned dataframes.
        If different behaviour is desired this method will have to be
        overridden as well in the metric implementation.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

            kwargs
                Configuration parameters as dictionary for the metric
                calculation.

        Returns
        -------
        Data frame containing the calculated metric(s) per sample.

        """

        metric_results = list()
        sample_names = np.asarray(matching["sample_name"].unique())

        for sample_name in sample_names:

            sample_annotation = annotation_data[
                annotation_data["sample_name"] == sample_name
            ]
            sample_prediction = prediction_data[
                prediction_data["sample_name"] == sample_name
            ]
            sample_matching = matching[matching["sample_name"] == sample_name]

            sample_metric = self.calc(
                annotation_data=sample_annotation,
                prediction_data=sample_prediction,
                matching=sample_matching,
                **kwargs
            )

            metric_results.append(sample_metric)

        if len(metric_results) == 0:
            return pd.DataFrame(
                data=None,
                columns=[
                    "total",
                ],
            )
        results = pd.concat(objs=metric_results, axis="index", ignore_index=True)
        results = results.set_index(sample_names)
        return results
