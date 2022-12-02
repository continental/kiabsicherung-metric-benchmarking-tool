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
Metric processor to compute the recall.

"""

import pandas as pd

from kia_mbt.kia_metrics.metric_processor import MetricProcessor
from kia_mbt.kia_metrics.number_of_true_positives import NumberOfTruePositives
from kia_mbt.kia_metrics.number_of_false_negatives import NumberOfFalseNegatives


class Recall(MetricProcessor):
    """
    This metric will be calculated on a per frame basis,
    as well as an average over the entire evaluated dataset.
    """

    def __init__(self):
        """
        Set metric identifier and name.

        """
        super().__init__(identifier=1028,
                         name='Recall',
                         calculate_per_sample=True)

        self._num_true_positive_processor = NumberOfTruePositives()
        self._num_false_negative_processor = NumberOfFalseNegatives()

    def calc(self,
             annotation_data: pd.DataFrame,
             prediction_data: pd.DataFrame,
             matching: pd.DataFrame,
             **kwargs) -> pd.DataFrame:
        """
        Calculate the recall given the matching.
        Recall = num_of_true_positives / (num_of_true_positives + num_of_false_negatives)

        Parameters
        ----------
            annotation_data_data : DataFrame
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
        The recall of the detections.

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)

        # compute recall = (num_tp / (num_tp + num_fn))
        num_tp = self._num_true_positive_processor.calc(annotation_data=annotation_data,
                                                        prediction_data=prediction_data,
                                                        matching=matching,
                                                        calculate_per_class=calculate_per_class)
        num_fn = self._num_false_negative_processor.calc(annotation_data=annotation_data,
                                                         prediction_data=prediction_data,
                                                         matching=matching,
                                                         calculate_per_class=calculate_per_class)
        recall = num_tp / (num_tp + num_fn)
        return recall
