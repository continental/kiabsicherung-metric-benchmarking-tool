# Copyright (c) 2022 Elektronische Fahrwerksysteme GmbH (www.efs-auto.com).
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
Metric processor to compute the number of true positives.

"""

import pandas as pd
from kia_mbt.kia_metrics.metric_processor import MetricProcessor


class NumberOfTruePositives(MetricProcessor):
    """
    Counting statistic. This metric will be calculated on a per frame basis,
    as well as a total over the entire evaluated dataset.
    """

    def __init__(self):
        """
        Set metric identifier and name.

        """
        super().__init__(identifier=1029,
                         name='Number of True Positives',
                         calculate_per_sample=True)

    def calc(self,
             annotation_data: pd.DataFrame,
             prediction_data: pd.DataFrame,
             matching: pd.DataFrame,
             **kwargs) -> pd.DataFrame:
        """
        Calculate the number of true positives in the matching.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

        Kwargs
        ------
            calculate_per_class : bool

        Returns
        -------
        Number of true positives.

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)

        # calculate the number of true positives
        if not calculate_per_class:
            num_true_positives = matching["confusion"].value_counts().get("tp", int(0))
            num_true_positives = pd.DataFrame(data=[num_true_positives, ], columns=["total", ])
        else:
            num_true_positives = self._calc_per_class(annotation_data=annotation_data,
                                                      prediction_data=prediction_data,
                                                      matching=matching)
        return num_true_positives

    def _calc_per_class(self,
                        annotation_data: pd.DataFrame,
                        prediction_data: pd.DataFrame,
                        matching: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of true positives in the matching per class.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

        Returns
        -------
        Number of true positives per class.

        """
        class_ids = list(matching["class_id"].unique())
        true_positives = dict()

        # total number of true positives
        true_positives["total"] = matching["confusion"].value_counts().get("tp", int(0))

        # number of true positives per class
        for class_id in class_ids:
            class_matching = matching[matching["class_id"] == class_id]
            num_class_tp = class_matching["confusion"].value_counts().get("tp", int(0))
            true_positives[class_id] = num_class_tp

        num_true_positives = pd.DataFrame(data=[true_positives, ])
        return num_true_positives
