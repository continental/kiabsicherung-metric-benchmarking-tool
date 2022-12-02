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
Metric processor to compute the mean intersection over union of true positives and false negatives.
False negatives are counted as 0.0 IOU.

"""

import pandas as pd

from kia_mbt.kia_metrics.metric_processor import MetricProcessor


class MeanIntersectionOverUnion(MetricProcessor):
    """
    Typically used for semantic segmentation is the mean of the IOU
    over all classes leading to the mIOU.
    """

    def __init__(self):
        """
        Set metric identifier and name.

        """
        super().__init__(identifier=1000,
                         name='Mean Intersection Over Union',
                         calculate_per_sample=True)

    def calc(self,
             annotation_data: pd.DataFrame,
             prediction_data: pd.DataFrame,
             matching: pd.DataFrame,
             **kwargs) -> pd.DataFrame:
        """
        Calculate the mean Intersection over Union of true positives and false negatives in the matching.

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
            iou_column_name : str

        Returns
        -------
        Mean intersection over union.

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)
        iou_column_name = kwargs.get("iou_column_name", "match_value")

        # calculate the mean intersection over union
        if not calculate_per_class:
            tp_fn_entries = matching[matching["confusion"].isin(["tp", "fn", ])]
            tp_fn_entries = tp_fn_entries.fillna(0.0)
            mean_iou = pd.DataFrame([tp_fn_entries[iou_column_name].mean(), ], columns=["total", ])
        else:
            mean_iou = self._calc_per_class(annotation_data=annotation_data,
                                            prediction_data=prediction_data,
                                            matching=matching,
                                            iou_column_name=iou_column_name)
        return mean_iou

    def _calc_per_class(self,
                        annotation_data: pd.DataFrame,
                        prediction_data: pd.DataFrame,
                        matching: pd.DataFrame,
                        iou_column_name) -> pd.DataFrame:
        """
        Calculate the mean intersection over union per class.

        Parameters
        ----------
            annotation_data_data : DataFrame
                Data frame containing the ground truth annotation data.

            prediction_data : DataFrame
                Data frame containing the prediction data.

            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

        Returns
        -------
        Mean intersection over union per class.

        """
        class_ids = list(matching["class_id"].unique())
        mean_iou = dict()

        tp_fn_entries = matching[matching["confusion"].isin(["tp", "fn", ])]
        tp_fn_entries = tp_fn_entries.fillna(0.0)
        mean_iou["total"] = tp_fn_entries[iou_column_name].mean()

        for class_id in class_ids:
            class_matching = matching[matching["class_id"] == class_id]

            tp_fn_entries = class_matching[class_matching["confusion"].isin(["tp", "fn", ])]
            tp_fn_entries = tp_fn_entries.fillna(0.0)
            mean_iou[class_id] = tp_fn_entries[iou_column_name].mean()

        class_mean_iou = pd.DataFrame(data=[mean_iou, ])
        return class_mean_iou
