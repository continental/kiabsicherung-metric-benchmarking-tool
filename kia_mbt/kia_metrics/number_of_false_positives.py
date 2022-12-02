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
Metric processor to compute the number of false positives.

"""

import pandas as pd
from kia_mbt.kia_metrics.metric_processor import MetricProcessor


class NumberOfFalsePositives(MetricProcessor):
    """
    Counting statistic. This metric will be calculated on a per frame basis,
    as well as a total over the entire evaluated dataset.
    """

    def __init__(self):
        """
        Set metric identifier and name.

        """
        super().__init__(identifier=1030,
                         name='Number of False Positives',
                         calculate_per_sample=True)

    def calc(self,
             annotation_data: pd.DataFrame,
             prediction_data: pd.DataFrame,
             matching: pd.DataFrame,
             **kwargs) -> pd.DataFrame:
        """
        Calculate the number of false positives in the matching.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the correlations between ground truth and
                the predictions.

        Kwargs
        ------
            calculate_per_class : bool

        Returns:
        --------
        Number of false positives.

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)

        # calculate the number of false positives
        if not calculate_per_class:
            num_false_positives = matching["confusion"].value_counts().get("fp", int(0))
            num_false_positives = pd.DataFrame(data=[num_false_positives, ], columns=["total", ])
        else:
            num_false_positives = self.calc_per_class(annotation_data=annotation_data,
                                                      prediction_data=prediction_data,
                                                      matching=matching)
        return num_false_positives

    def calc_per_class(self,
                       annotation_data: pd.DataFrame,
                       prediction_data: pd.DataFrame,
                       matching: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of false positives in the matching per class.

        Parameters
        ----------
            annotation_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the correlations between ground truth and
                the predictions.

        Returns
        -------
        Number of false positives per class.

        """
        class_ids = matching["class_id"].unique()
        false_positives = dict()

        # total number of false positives in sample
        false_positives["total"] = matching["confusion"].value_counts().get("fp", int(0))

        # number of false positives per sample per class
        for class_id in class_ids:
            class_matching = matching[matching["class_id"] == class_id]
            num_class_fp = class_matching["confusion"].value_counts().get("fp", int(0))
            false_positives[class_id] = num_class_fp

        num_false_positives = pd.DataFrame(data=[false_positives, ])
        return num_false_positives
