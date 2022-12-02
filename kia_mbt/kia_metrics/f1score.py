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
Metric processor for F1-score.

"""

import pandas as pd
from kia_mbt.kia_metrics.metric_processor import MetricProcessor

from kia_mbt.kia_metrics.number_of_true_positives import NumberOfTruePositives
from kia_mbt.kia_metrics.number_of_false_positives import NumberOfFalsePositives
from kia_mbt.kia_metrics.number_of_false_negatives import NumberOfFalseNegatives


class F1Score(MetricProcessor):
    """
    F1-Score: 2*true_positives / (2 * true_positives + false_positives + false_negatives).
    As with mIou usually the mean over all classes is used.
    """

    def __init__(self):
        """
        Set metric identifier and name.

        """
        super().__init__(identifier=1001,
                         name='F1-Score',
                         calculate_per_sample=True)

        self._num_true_positive_processor = NumberOfTruePositives()
        self._num_false_positive_processor = NumberOfFalsePositives()
        self._num_false_negative_processor = NumberOfFalseNegatives()

    def calc(self,
             annotation_data: pd.DataFrame,
             prediction_data: pd.DataFrame,
             matching: pd.DataFrame,
             **kwargs) -> pd.DataFrame:
        """
        Calculate the F1-score given the matching.

        F1-Score = (2 * #tp) / (2 * #tp + #fp + #fn)
         = (2 * precision * recall) / (precision + recall)

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

        Returns
        -------
        The F1-score of the detections.

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)

        # calculate the F1-score
        num_tp = self._num_true_positive_processor.calc(annotation_data=annotation_data,
                                                        prediction_data=prediction_data,
                                                        matching=matching,
                                                        calculate_per_class=calculate_per_class)
        num_fp = self._num_false_positive_processor.calc(annotation_data=annotation_data,
                                                         prediction_data=prediction_data,
                                                         matching=matching,
                                                         calculate_per_class=calculate_per_class)
        num_fn = self._num_false_negative_processor.calc(annotation_data=annotation_data,
                                                         prediction_data=prediction_data,
                                                         matching=matching,
                                                         calculate_per_class=calculate_per_class)
        f1_score = (2.0 * num_tp) / (2.0 * num_tp + num_fp + num_fn)
        return f1_score
