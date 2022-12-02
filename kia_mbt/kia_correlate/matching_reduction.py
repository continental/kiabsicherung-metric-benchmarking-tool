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
Reduce m-to-n matching to 1-to-1 matching.

"""

import pandas as pd
import numpy as np


class MatchingReduction():
    """
    Class for reducing m-to-n bounding-box matching to 1-to-1 matching.
    """

    def reduce_to_exclusive(self,
                            matching: pd.DataFrame,
                            confidence_column_name: str = "confidence",
                            iou_column_name: str = "match_value",
                            sort_output: bool = True) -> pd.DataFrame:
        """
        Reduce m-to-n matching dataframe to 1-to-1.

        Parameters
        ----------
            matching: pandas.DataFrame
                Data frame containing m-to-n bounding-box matching.

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
            matching_exclusive: pandas.DataFrame
                Data frame containing 1-to-1 bounding-box matching.

        """
        exclusive_samples = list()

        # get all sample names
        sample_names = sorted(list(matching["sample_name"].unique()))

        for sample_name in sample_names:
            # filter correlation data by sample name
            sample_matching = matching[matching["sample_name"] == sample_name]

            tp_sample_data = sample_matching[sample_matching["confusion"] == "tp"]
            fp_keep = sample_matching[sample_matching["confusion"] == "fp"]
            fn_keep = sample_matching[sample_matching["confusion"] == "fn"]

            tp_sample_sorted = tp_sample_data.sort_values(by=[confidence_column_name, iou_column_name],
                                                          axis="index",
                                                          ascending=(False, False),
                                                          inplace=False)

            tp_annotation_ids = set(tp_sample_sorted.annotation_index.unique())
            tp_detection_ids = set(tp_sample_sorted.detection_index.unique())

            # drop duplicates in detection_index column to get entry with largest iou overlap
            possible_matches = tp_sample_sorted.drop_duplicates(subset="detection_index")

            # drop duplicates in annotation_index column to get entry with largest iou overlap
            tp_keep = possible_matches.drop_duplicates(subset="annotation_index")
            keep_annotation_ids = set(tp_keep.annotation_index)
            keep_detection_ids = set(tp_keep.detection_index)

            # get differences in annotation_index entries to detect new false negatives
            new_fn_annotation_ids = list(tp_annotation_ids - keep_annotation_ids)

            # get differences in detection_index entries to detect new false positives
            new_fp_detection_ids = list(tp_detection_ids - keep_detection_ids)

            # new false negatives
            fn_update = tp_sample_sorted.loc[tp_sample_sorted["annotation_index"].isin(new_fn_annotation_ids)]
            fn_update = fn_update.drop_duplicates(subset="annotation_index", inplace=False)
            fn_update.loc[:, "detection_index"] = None
            fn_update.loc[:, "confusion"] = "fn"
            fn_update.loc[:, "match_value"] = np.nan
            fn_update.loc[:, "confidence"] = np.nan

            # new false positives
            fp_update = tp_sample_sorted[tp_sample_sorted["detection_index"].isin(new_fp_detection_ids)]
            fp_update = fp_update.drop_duplicates(subset="detection_index", inplace=False)
            fp_update.loc[:, "annotation_index"] = None
            fp_update.loc[:, "confusion"] = "fp"
            fp_update.loc[:, "match_value"] = np.nan

            # concatenate result
            sample_exclusive = pd.concat(objs=[tp_keep, fp_keep, fp_update, fn_keep, fn_update],
                                         axis='index',
                                         ignore_index=True,
                                         verify_integrity=True,
                                         copy=True)
            # re-sort the exclusive sample to get original ordering
            if sort_output:
                sample_exclusive = sample_exclusive.sort_values(
                    by=["class_id", "confusion", "annotation_index", "detection_index"],
                    axis="index",
                    ascending=[True, False, True, True],
                    inplace=False,
                    na_position="last",
                    ignore_index=True)
            exclusive_samples.append(sample_exclusive)

        if len(exclusive_samples) == 0:
            matching_cols = ['sample_name', 'annotation_index', 'detection_index',
                             'confusion', 'class_id', iou_column_name, confidence_column_name]
            return pd.DataFrame(data=None, columns=matching_cols)

        matching_exclusive = pd.concat(objs=exclusive_samples,
                                       axis='index',
                                       ignore_index=True,
                                       verify_integrity=True,
                                       copy=True)
        return matching_exclusive
