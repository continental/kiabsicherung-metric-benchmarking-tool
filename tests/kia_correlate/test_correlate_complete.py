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
Unit test BoxCorrelator complete matching.

"""

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from kia_mbt.kia_correlate.box_correlator import BoxCorrelator
from tests.kia_correlate.conftest import get_empty_data
from tests.kia_correlate.conftest import get_test_data
from tests.kia_correlate.conftest import get_test_data_clipped_boxes


def test_correlate_complete_init():
    """
    Test box_correlator complete initialization.
    """
    # arrange
    box_correlator_1 = BoxCorrelator()
    box_correlator_2 = BoxCorrelator(threshold=0.1,
                                     matching_type="complete",
                                     clip_truncated_boxes=False)
    box_correlator_3 = BoxCorrelator(threshold=0.1,
                                     matching_type="complete",
                                     clip_truncated_boxes=False,
                                     clip_x=(0.0, 1440.0),
                                     clip_y=(0.0, 900.0))
    box_correlator_4 = BoxCorrelator(threshold=0.1,
                                     matching_type="complete",
                                     clip_truncated_boxes=True,
                                     clip_x=(0.0, 1440.0),
                                     clip_y=(0.0, 900.0))

    with pytest.raises(RuntimeError):
        box_correlator_5 = BoxCorrelator(threshold=0.1,
                                         matching_type="custom")

    # assert 1
    assert box_correlator_1._threshold == 0.5, "wrong _threshold"
    assert box_correlator_1._matching_type == "complete", "wrong _matching_type"
    assert box_correlator_1._clip_truncated_boxes is True, "wrong _clip_truncated_boxes"
    assert box_correlator_1._confidence_col == "confidence", "wrong _confidence_col"
    assert box_correlator_1._annotation_bb_center_col == "center"
    assert box_correlator_1._annotation_bb_size_col == "size"
    assert box_correlator_1._detection_bb_center_col == "center"
    assert box_correlator_1._detection_bb_size_col == "size"
    assert box_correlator_1._clip_x == approx((0.0, 1920.0))
    assert box_correlator_1._clip_y == approx((0.0, 1280.0))

    # assert 2
    assert box_correlator_2._threshold == 0.1, "wrong _threshold"
    assert box_correlator_2._matching_type == "complete", "wrong _matching_type"
    assert box_correlator_2._clip_truncated_boxes is False, "wrong _clip_truncated_boxes"
    assert box_correlator_2._clip_x == (-np.inf, np.inf)
    assert box_correlator_2._clip_y == (-np.inf, np.inf)

    # assert 3
    assert box_correlator_3._threshold == 0.1, "wrong _threshold"
    assert box_correlator_3._matching_type == "complete", "wrong matching_type"
    assert box_correlator_3._clip_truncated_boxes is False, "wrong _clip_truncated_boxes"
    assert box_correlator_3._clip_x == (-np.inf, np.inf)
    assert box_correlator_3._clip_y == (-np.inf, np.inf)

    # assert 4
    assert box_correlator_4._threshold == 0.1, "wrong _threshold"
    assert box_correlator_4._matching_type == "complete", "wrong _matching_type"
    assert box_correlator_4._clip_truncated_boxes is True, "wrong _clip_truncated_boxes"
    assert box_correlator_4._clip_x == approx((0.0, 1440.0))
    assert box_correlator_4._clip_y == approx((0.0, 900.0))


def test_correlate_complete_empty_data():
    """
    Test case with empty data.
    """
    # arrange
    box_correlator = BoxCorrelator(threshold=0.5,
                                   matching_type="complete")
    annotation_data, prediction_data, _ = get_empty_data()
    # act
    ans = box_correlator(annotation_data=annotation_data,
                         detection_data=prediction_data)
    # assert
    assert ans.empty is True


def test_correlate_complete():
    """
    Test case with true positives, false positives and false negatives.
    """
    # arrange
    box_correlator = BoxCorrelator(threshold=0.5,
                                   matching_type="complete")
    annotation_data, prediction_data, _ = get_test_data()
    # act
    ans = box_correlator(annotation_data=annotation_data,
                         detection_data=prediction_data)
    confusion_counts = ans["confusion"].value_counts()
    # assert
    assert ans.shape[0] == 9
    assert ans.shape[1] == 7
    assert confusion_counts["tp"] == 4
    assert confusion_counts["fp"] == 2
    assert confusion_counts["fn"] == 3


def test_correlate_complete_clipping():
    """
    Test case with clipped bounding boxes.
    """
    # arrange
    box_correlator = BoxCorrelator(threshold=0.6,
                                   matching_type="complete")
    annotation_data, prediction_data = get_test_data_clipped_boxes()
    # act
    ans = box_correlator(annotation_data=annotation_data,
                         detection_data=prediction_data)
    # assert
    assert isinstance(ans, pd.DataFrame)
    assert len(ans) == 1
    assert ans["sample_name"][0] == annotation_data["sample_name"][0]
    assert ans["annotation_index"][0] == annotation_data.index[0]
    assert ans["detection_index"][0] == prediction_data.index[0]
    assert ans["match_value"][0] == approx(1.0)
