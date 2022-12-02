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
Applying bounding-box matching thresholds.

"""

import pandas as pd
import numpy as np


class MatchingThreshold():
    """
    Class for applying updated bounding-box matching thresholds.
    """

    def apply_iou_threshold(self,
                            matching: pd.DataFrame,
                            iou_threshold: float,
                            confidence_column_name: str = "confidence",
                            iou_column_name: str = "match_value",
                            sort_output: bool = True) -> pd.DataFrame:
        """Apply updated IOU threshold to m-to-n matching dataframe.

        Parameters
        ----------
            matching : pandas.DataFrame
                Data frame containing bounding-box matching.
                Has to be computed with an initial threshold smaller than
                iou_threshold, otherwise nothing will be done here.

            iou_threshold : float
                New iou_threshold. Has to be larger than current threshold in matching.

            confidence_column_name : str
                Optional: defaults to "confidence"
                Name of the confidence score column in matching DataFrame.

            iou_column_name : str
                Optional: defaults to "match_value"
                Name of the IOU column in matching DataFrame.

            sort_output : bool
                Whether to sort the output for readability.

        Returns
        -------
            matching_updated: pandas.DataFrame
                Data frame containing updated bounding-box matching.

        """
        matching_list = list()

        # get all sample names
        sample_names = sorted(list(matching["sample_name"].unique()))

        for sample_name in sample_names:
            # filter matching data by sample name
            sample_matching = matching[matching["sample_name"] == sample_name]

            # false positives and false negatives
            fp_keep = sample_matching[sample_matching["confusion"] == "fp"]
            fn_keep = sample_matching[sample_matching["confusion"] == "fn"]

            # true positives with score above / below threshold
            tp_data = sample_matching[sample_matching["confusion"] == "tp"]
            tp_keep = tp_data[tp_data[iou_column_name] >= iou_threshold]
            tp_check = tp_data[tp_data[iou_column_name] < iou_threshold]

            annotation_ids_keep = set(tp_keep["annotation_index"])
            detection_ids_keep = set(tp_keep["detection_index"])

            # check if removed annotation indices become false negatives
            fn_update = tp_check[~tp_check["annotation_index"].isin(annotation_ids_keep)]
            fn_update = fn_update.drop_duplicates(subset="annotation_index")
            fn_update.loc[:, "detection_index"] = None
            fn_update.loc[:, "confusion"] = "fn"
            fn_update.loc[:, iou_column_name] = np.nan
            fn_update.loc[:, "confidence"] = np.nan

            # check if removed detection indices become false positives
            fp_update = tp_check[~tp_check["detection_index"].isin(detection_ids_keep)]
            fp_update = fp_update.drop_duplicates(subset="detection_index")
            fp_update.loc[:, "annotation_index"] = None
            fp_update.loc[:, "confusion"] = "fp"
            fp_update.loc[:, iou_column_name] = np.nan

            # concatenate result
            sample_updated = pd.concat(objs=[tp_keep, fp_keep, fp_update, fn_keep, fn_update],
                                       axis='index',
                                       ignore_index=True,
                                       verify_integrity=True,
                                       copy=True)
            # re-sort the updated sample to get original ordering
            if sort_output:
                sample_updated = sample_updated.sort_values(by=["class_id", "confusion", "annotation_index", "detection_index"],
                                                            axis="index",
                                                            ascending=[True, False, True, True],
                                                            inplace=False,
                                                            na_position="last",
                                                            ignore_index=True)
            matching_list.append(sample_updated)

        if len(matching_list) == 0:
            matching_cols = ['sample_name', 'annotation_index', 'detection_index',
                             'confusion', 'class_id', iou_column_name, confidence_column_name]
            return pd.DataFrame(data=None, columns=matching_cols)

        matching_updated = pd.concat(objs=matching_list,
                                     axis='index',
                                     ignore_index=True,
                                     verify_integrity=True,
                                     copy=True)
        return matching_updated

    def apply_confidence_threshold(self,
                                   matching: pd.DataFrame,
                                   confidence_threshold: float,
                                   confidence_column_name: str = "confidence",
                                   iou_column_name: str = "match_value",
                                   sort_output=True) -> pd.DataFrame:
        """Apply confidence threshold to complete m-to-n matching dataframe.

        Parameters
        ----------
            matching: pandas.DataFrame
                Data frame containing bounding-box matching.
                Has to be computed with initial an threshold smaller than
                confidence_threshold, otherwise nothing will be done here.

            confidence_threshold: float
                New confidence threshold. Has to be larger than current threshold in matching.

            confidence_column_name: str
                Optional: defaults to "confidence"
                Name of the confidence column name in matching DataFrame.

            iou_column_name : str
                Optional: defaults to "match_value"
                Name of the IOU column in matching DataFrame.

            sort_output : bool
                Whether to sort the output for readability.

        Returns
        -------
            matching_updated: pandas.DataFrame
                Data frame containing updated bounding-box matching.

        """
        matching_list = list()

        # get all sample names
        sample_names = sorted(list(matching.sample_name.unique()))

        for sample_name in sample_names:
            # filter matching data by sample name
            sample_matching = matching[matching["sample_name"] == sample_name]

            # false positives and false negatives
            fn_keep = sample_matching[sample_matching["confusion"] == "fn"]

            fp_check = sample_matching[sample_matching["confusion"] == "fp"]
            fp_keep = fp_check[fp_check[confidence_column_name] >= confidence_threshold]

            # true positives with score above / below threshold
            tp_data = sample_matching[sample_matching["confusion"] == "tp"]
            tp_keep = tp_data[tp_data[confidence_column_name] >= confidence_threshold]
            tp_check = tp_data[tp_data[confidence_column_name] < confidence_threshold]

            annotation_ids_keep = set(tp_keep["annotation_index"])

            # check if removed annotation matches become false negatives
            fn_update = tp_check[~tp_check["annotation_index"].isin(annotation_ids_keep)]
            fn_update = fn_update.drop_duplicates(subset="annotation_index")
            fn_update.loc[:, "detection_index"] = None
            fn_update.loc[:, "confusion"] = "fn"
            fn_update.loc[:, iou_column_name] = np.nan
            fn_update.loc[:, confidence_column_name] = np.nan

            # concatenate result
            sample_updated = pd.concat(objs=[tp_keep, fp_keep, fn_keep, fn_update],
                                       axis='index',
                                       ignore_index=True,
                                       verify_integrity=True,
                                       copy=True)
            # re-sort the updated sample to get the original ordering
            if sort_output:
                sample_updated = sample_updated.sort_values(
                    by=["class_id", "confusion", "annotation_index", "detection_index"],
                    axis="index",
                    ascending=[True, False, True, True],
                    inplace=False,
                    na_position="last",
                    ignore_index=True)
            matching_list.append(sample_updated)

        if len(matching_list) == 0:
            matching_cols = ['sample_name', 'annotation_index', 'detection_index',
                             'confusion', 'class_id', iou_column_name, confidence_column_name]
            return pd.DataFrame(data=None, columns=matching_cols)

        matching_updated = pd.concat(objs=matching_list,
                                     axis='index',
                                     ignore_index=True,
                                     verify_integrity=True,
                                     copy=True)
        return matching_updated
