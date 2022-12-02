# Copyright (c) 2022 LZR, Bergische Universität Wuppertal (https://www.lzr.uni-wuppertal.de/de/).
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
Metric processor to compute the VOC mAP.

"""

import pandas as pd
import numpy as np

from kia_mbt.kia_metrics.metric_processor import MetricProcessor
from kia_mbt.kia_metrics.precision_recall_curve import PrecisionRecallCurve


class VocMAP(MetricProcessor):
    """
    The average precision (AP) for a specific class is calculated as the area under the precision-recall curve,
    where precision and recall for a given class are plotted for decreasing confidence thresholds.
    This way the metric balances false positive and false negative rates to a certain extent.

    It can be used to calculate the mAP if predictions for multiple classes were present.
    Mean average precision (mAP) computes the mean of the per-class average precisions.
    """

    def __init__(self):
        """
        Set metric identifier and name.
        """
        super().__init__(identifier=1003,
                         name='VOC mAP',
                         calculate_per_sample=True)

        self.prec_recall_processor = PrecisionRecallCurve()

    def calc(self,
             annotation_data: pd.DataFrame,
             prediction_data: pd.DataFrame,
             matching: pd.DataFrame,
             **kwargs) -> pd.DataFrame:
        """
        Calculate the AP for each class given the matching and the mean AP over classes (mAP).
        Average precision (AP) is calculated as the area under the Precision-Recall Curve.

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
                Whether to output AP for each class.
            ap_integration_mode : str
                Supported modes are "11point", "exact"
            confidence_col : str
                Name of the confidence value column

        Returns
        -------
        The mAP-score of the detections.

        """
        # extract kwargs
        calculate_per_class = kwargs.get("calculate_per_class", True)
        ap_integration_mode = kwargs.get("ap_integration_mode", "11point")
        confidence_col = kwargs.get("confidence_col", "confidence")
        eps = kwargs.get("eps", 0.0)

        # initialize results dict for AP scores
        ap_scores = {}

        # iterate over classes in matching table
        class_ids = matching["class_id"].unique()

        for class_id in class_ids:
            # class_annotation = annotation_data[annotation_data["class_id"] == class_id]
            # class_prediction = prediction_data[prediction_data["class_id"] == class_id]
            class_matching = matching[matching["class_id"] == class_id]

            # calculate precision-recall curve
            recall, precision = self.prec_recall_processor.prec_recall_curve(
                matching=class_matching,
                confidence_col=confidence_col,
            )
            # calculate average precision, mean recall, mean precision
            if ap_integration_mode == '11point':
                ap_score = self.voc_ap_2007(
                    recall=recall,
                    precision=precision,
                    eps=eps
                )

            elif ap_integration_mode == 'exact':
                ap_score = self.voc_ap_exact(
                    recall=recall,
                    precision=precision
                )

            ap_scores[class_id] = ap_score

        # compute mean average precision (average over classes)
        map_score = 0.0
        n_non_nan_scores = 0
        for val in ap_scores.values():
            map_score += val
            n_non_nan_scores += 1
        if n_non_nan_scores != 0:
            map_score = map_score / n_non_nan_scores
        else:
            map_score = np.nan

        # format output as dataframe
        if calculate_per_class:
            ap_scores["mAP"] = map_score
            ans = pd.DataFrame(data=[ap_scores, ])
        else:
            ans = pd.DataFrame(data=[{"mAP": map_score}])
        return ans

    def voc_ap_2007(self,
                    recall: np.ndarray,
                    precision: np.ndarray,
                    eps: float = 0.0) -> float:
        """
        Use the VOC 07 11 point method for AP.
        Used just 11 points, equally spaced between 0.0 and 1.0 to estimate
        the area under the precision recall curve.

        Parameters
        ----------
            recall : np.ndarray
                Array containing recall values.
            precision : np.ndarray
                Array containing precision values.
            eps : float

        Returns
        -------
            ap_score : float
                Average precision score.

        """
        ap_score = 0.0
        for thres in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= thres) == 0:
                p_add = 0.0
            else:
                p_add = np.max(precision[recall >= (thres - eps)])
            ap_score = ap_score + p_add
        ap_score = ap_score / 11.0
        return ap_score

    def voc_ap_exact(self,
                     recall: np.ndarray,
                     precision: np.ndarray) -> float:
        """
        Use the VOC 10-12 method for AP.
        Here the exact area under the precision recall after removing the zigzags
        is computed, no interpolation.

        Parameters
        ----------
            recall : np.ndarray
                Array containing recall values.
            precision : np.ndarray
                Array containing precision values.

        Returns
        -------
            ap_score : float
                Average precision score.

        """
        if np.isnan(precision).any():
            return np.nan
        if np.isnan(recall).any():
            return np.nan
        # --- Official matlab code VOC2012---
        # mrec=[0 ; rec ; 1];
        # mpre=[0 ; prec ; 0];
        # for i=numel(mpre)-1:-1:1
        #         mpre(i)=max(mpre(i),mpre(i+1));
        # end
        # i=find(mrec(2:end)~=mrec(1:end-1))+1;
        # ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # This part makes the precision monotonically decreasing
        #    (goes from the end to the beginning)
        #    matlab: for i=numel(mpre)-1:-1:1
        #                mpre(i)=max(mpre(i),mpre(i+1));
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # This part creates a list of indexes where the recall changes
        #    matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        idx_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                idx_list.append(i) # if it was matlab would be i + 1

        # The Average Precision (AP) is the area under the curve
        #    (numerical integration)
        #    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        ap_score = 0.0
        for i in idx_list:
            ap_score += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap_score
