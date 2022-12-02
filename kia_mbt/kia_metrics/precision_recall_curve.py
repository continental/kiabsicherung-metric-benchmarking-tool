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
Metric processor to compute precision-recall curves.

"""

from typing import Tuple
import pandas as pd
import numpy as np

from kia_mbt.kia_metrics.metric_processor import MetricProcessor


class PrecisionRecallCurve(MetricProcessor):
    """
    Precision-Recall Curve.
    The precision-recall curve is computed from a method's ranked output.
    """

    def __init__(self):
        """
        Set metric identifier and name.
        """
        super().__init__(
            identifier=1040, name="Precision-Recall Curve", calculate_per_sample=True
        )

    def calc(
        self,
        annotation_data: pd.DataFrame,
        prediction_data: pd.DataFrame,
        matching: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute the precision-recall curve(s).

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
                Whether to compute the precision-recall curve for each class.
            confidence_col : str
                Name of the confidence value column.

        Returns
        -------
        Data frame containing the calculated metric(s).

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)
        confidence_col = kwargs.get("confidence_col", "confidence")

        # compute precision-recall curves
        if not calculate_per_class:
            rec, prec = self.prec_recall_curve(
                matching=matching, confidence_col=confidence_col
            )
            recall_prec_curves = pd.DataFrame(
                data=[[(list(rec), list(prec))]],
                columns=[
                    "total",
                ],
            )
        else:
            recall_prec_curves = self.calc_per_class(
                annotation_data=annotation_data,
                prediction_data=prediction_data,
                matching=matching,
                confidence_col=confidence_col,
            )
        return recall_prec_curves

    def calc_per_class(
        self,
        annotation_data: pd.DataFrame,
        prediction_data: pd.DataFrame,
        matching: pd.DataFrame,
        confidence_col: str,
    ) -> pd.DataFrame:
        """
        Compute precision-recall curve from the matching per class.

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
        Precision-recall curve per class.

        """
        class_ids = matching["class_id"].unique()
        prec_recall = dict()

        # total average precision-recall
        rec, prec = self.prec_recall_curve(
            matching=matching, confidence_col=confidence_col
        )
        prec_recall["total"] = [(list(rec), list(prec))]

        # precision-recall per class
        for class_id in class_ids:
            class_matching = matching[matching["class_id"] == class_id]
            rec, prec = self.prec_recall_curve(
                matching=class_matching, confidence_col=confidence_col
            )
            prec_recall[class_id] = [
                (list(rec), list(prec)),
            ]

        prec_recall_curves = pd.DataFrame(data=prec_recall)
        return prec_recall_curves

    def prec_recall_curve(
        self,
        matching: pd.DataFrame,
        confidence_col: str = "confidence",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve.

        Parameters
        ----------
            matching : DataFrame
                Data frame containing the matching between ground truth and
                the predictions.

            confidence_col : str
                Name of the confidence value column.

        Returns
        -------
        Data frame containing the calculated metric(s).

        """
        # get total number of ground-truth instances
        matching_counts = matching["confusion"].value_counts()
        tot_num_tp = matching_counts.get("tp", int(0))
        tot_num_fn = matching_counts.get("fn", int(0))
        num_gt_instances = tot_num_tp + tot_num_fn

        # remove fn to get list of predictions only
        matching_preds = matching[matching["confusion"].isin(["tp", "fp"])]

        # sort predictions by confidence in descending order
        matching_sorted = matching_preds.sort_values(
            by=confidence_col, axis="index", ascending=False
        )

        # create binary lists with positions of tp and fp
        cntr_tp = np.array((matching_sorted["confusion"] == "tp") * 1)
        if len(cntr_tp) == 0:  # no true positives in matching
            cntr_tp = np.zeros(shape=(1,))

        cntr_fp = np.array((matching_sorted["confusion"] == "fp") * 1)

        # increasing counters for all predictions with higher confidence
        cntr_tp = np.cumsum(cntr_tp)
        cntr_fp = np.cumsum(cntr_fp)

        # calculate recall at "confidence threshold"
        if num_gt_instances != 0:
            rec = cntr_tp / float(num_gt_instances)
        else:
            rec = np.asarray(
                [
                    np.nan,
                ]
            )

        # calculate precision at "confidence threshold"
        if len(matching_sorted) != 0:
            prec = cntr_tp / np.maximum(cntr_tp + cntr_fp, np.finfo(np.float64).eps)
        else:
            prec = np.asarray(
                [
                    np.nan,
                ]
            )

        return rec, prec  # x, y - in precision-recall curve
