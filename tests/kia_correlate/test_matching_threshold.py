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
Unit test matching_threshold.

"""

import numpy as np
import pandas as pd

from kia_mbt.kia_correlate.matching_threshold import MatchingThreshold
from tests.kia_correlate.conftest import get_empty_data
from tests.kia_correlate.conftest import get_test_data_single_tp
from tests.kia_correlate.conftest import get_test_data_single_fp_fn
from tests.kia_correlate.conftest import get_test_data_tp_with_alternative_matches

#######################
# IOU threshold tests #
#######################


def test_apply_iou_threshold_empty_data():
    """
    Test application of iou_threshold with empty input.
    """
    # arrange
    threshold = MatchingThreshold()
    _, _, matching = get_empty_data()
    # act
    ans = threshold.apply_iou_threshold(matching=matching,
                                        iou_threshold=0.5)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"


def test_apply_iou_threshold_single_true_positive():
    """
    Test application of iou_threshold with single true positive.
    """
    # arrange
    threshold = MatchingThreshold()
    matching = get_test_data_single_tp()
    # act
    ans1 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.4)
    ans2 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.6)
    # assert 1
    comp1 = ans1.compare(matching)
    assert comp1.empty is True, "comparison is not empty"
    # assert 2
    assert len(ans2) == 2
    assert ans2["confusion"].equals(pd.Series(["fp", "fn"]))
    assert ans2["detection_index"][0] == matching["detection_index"][0]
    assert pd.isna(ans2["detection_index"][1])
    assert pd.isna(ans2["annotation_index"][0])
    assert ans2["annotation_index"][1] == matching["annotation_index"][0]
    assert pd.isna(ans2["match_value"][0])
    assert pd.isna(ans2["match_value"][1])


def test_apply_iou_threshold_single_false_positive_false_negative():
    """
    Test application of iou_threshold with single false positive and false negative.
    """
    # arrange
    threshold = MatchingThreshold()
    matching = get_test_data_single_fp_fn()
    # act
    ans1 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.4)
    ans2 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.6)
    # assert
    assert ans1.compare(matching).empty is True
    assert ans2.compare(matching).empty is True


def test_apply_iou_threshold_tp_with_alternative_match():
    """
    Test application of iou_threshold for matching data with multiple true positives.
    """
    # arrange
    threshold = MatchingThreshold()
    matching = get_test_data_tp_with_alternative_matches()
    # act
    ans1 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.2)
    ans2 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.4)
    ans3 = threshold.apply_iou_threshold(matching=matching,
                                         iou_threshold=0.6)
    # assert 1
    assert ans1.compare(matching).empty is True
    # assert 2
    assert len(ans2) == 4, "wrong number of rows"
    np.array_equal(a1=ans2.values,
                   a2=matching[[True, False, True, True, True, ]].values)
    # assert 3
    assert len(ans3) == 4, "wrong number of rows"
    np.array_equal(a1=ans3[[True, True, False, False, ]].values,
                   a2=matching[[True, False, False, True, False, ]].values)

    assert ans3["confusion"][2] == "fp"
    assert pd.isna(ans3["annotation_index"][2])
    assert ans3["detection_index"][2] == matching["detection_index"][2]
    assert pd.isna(ans3["match_value"][2])

    assert ans3["confusion"][3] == "fn"
    assert ans3["annotation_index"][3] == matching["annotation_index"][4]
    assert pd.isna(ans3["detection_index"][3])
    assert pd.isna(ans3["match_value"][3])


##############################
# confidence threshold tests #
##############################

def test_confidence_threshold_empty_data():
    """
    Test application of confidence_threshold with empty input.
    """
    # arrange
    threshold = MatchingThreshold()
    _, _, matching = get_empty_data()
    # act
    ans = threshold.apply_confidence_threshold(matching=matching,
                                               confidence_threshold=0.5)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"


def test_apply_confidence_threshold_single_true_positive():
    """
    Test application of confidence_threshold with single true positive.
    """
    # arrange
    threshold = MatchingThreshold()
    matching = get_test_data_single_tp()
    # act
    ans1 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.4)
    ans2 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.6)
    # assert 1
    comp1 = ans1.compare(matching)
    assert comp1.empty is True, "comparison is not empty"
    # assert 2
    assert len(ans2) == 1
    assert ans2["sample_name"][0] == matching["sample_name"][0]
    assert ans2["annotation_index"][0] == matching["annotation_index"][0]
    assert pd.isna(ans2["detection_index"][0])
    assert ans2["confusion"][0] == "fn"
    assert ans2["class_id"][0] == matching["class_id"][0]
    assert pd.isna(ans2["match_value"][0])
    assert pd.isna(ans2["confidence"][0])


def test_apply_confidence_threshold_single_false_positive_false_negative():
    """
    Test application of confidence_threshold with single false positive and false negative.
    """
    # arrange
    threshold = MatchingThreshold()
    matching = get_test_data_single_fp_fn()
    # act
    ans1 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.4)
    ans2 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.6)
    # assert 1
    assert ans1.compare(matching).empty is True
    # assert 2
    assert len(ans2) == 1
    assert ans2.loc[0].equals(matching.loc[1])


def test_apply_confidence_threshold_tp_with_different_match():
    """
    Test application of confidence_threshold for matching data with multiple true positives.
    """
    # arrange
    threshold = MatchingThreshold()
    matching = get_test_data_tp_with_alternative_matches()
    # act
    ans1 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.2)
    ans2 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.4)
    ans3 = threshold.apply_confidence_threshold(matching=matching,
                                                confidence_threshold=0.6)
    # assert 1
    assert ans1.compare(matching).empty is True
    # assert 2
    assert len(ans2) == 4, "wrong number of rows"
    np.array_equal(a1=ans2.values,
                   a2=matching[[True, False, True, True, True, ]].values)
    # assert 3
    assert len(ans3) == 3, "wrong number of rows"
    np.array_equal(a1=ans3[[True, True, False, ]].values,
                   a2=matching[[True, False, False, True, False, ]].values)

    assert ans3["sample_name"][2] == matching["sample_name"][4]
    assert ans3["annotation_index"][2] == matching["annotation_index"][4]
    assert pd.isna(ans3["detection_index"][2])
    assert ans3["confusion"][2] == "fn"
    assert pd.isna(ans3["match_value"][2])
    assert pd.isna(ans3["confidence"][2])
