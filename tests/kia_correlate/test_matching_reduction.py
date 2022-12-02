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
Unit test matching_reduction.

"""

import numpy as np
import pandas as pd

from kia_mbt.kia_correlate.matching_reduction import MatchingReduction
from tests.kia_correlate.conftest import get_empty_data
from tests.kia_correlate.conftest import get_test_data_single_tp
from tests.kia_correlate.conftest import get_test_data_single_fp_fn
from tests.kia_correlate.conftest import get_test_data_tp_with_alternative_matches
from tests.kia_correlate.conftest import get_test_data_three_true_positives_one_annotation
from tests.kia_correlate.conftest import get_test_data_three_true_positives_one_detection


def test_reduction_empty_data():
    """
    Test matching reduction with empty input.
    """
    # arrange
    reduction = MatchingReduction()
    _, _, matching = get_empty_data()
    # act
    ans = reduction.reduce_to_exclusive(matching=matching)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"


def test_reduction_single_true_positive():
    """
    Test matching reduction with single true positive.
    """
    # arrange
    reduction = MatchingReduction()
    matching = get_test_data_single_tp()
    # act
    ans = reduction.reduce_to_exclusive(matching=matching)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans.compare(matching).empty is True


def test_reduction_single_false_positive_false_negative():
    """
    Test matching reduction with single false positive and false negative.
    """
    # arrange
    reduction = MatchingReduction()
    matching = get_test_data_single_fp_fn()
    # act
    ans = reduction.reduce_to_exclusive(matching=matching)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans.compare(matching).empty is True


def test_reduction_with_alternative_match():
    """
    Test matching reduction with multiple alternative true positives.
    """
    # arrange
    reduction = MatchingReduction()
    matching = get_test_data_tp_with_alternative_matches()
    # act
    ans = reduction.reduce_to_exclusive(matching=matching)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 4
    np.array_equal(a1=ans[[True, True, False, False, ]].values,
                   a2=matching[[True, False, False, True, False, ]].values)

    assert ans["confusion"][2] == "fp"
    assert pd.isna(ans["annotation_index"][2])
    assert ans["detection_index"][2] == matching["detection_index"][2]
    assert pd.isna(ans["match_value"][2])

    assert ans["confusion"][3] == "fn"
    assert ans["annotation_index"][3] == matching["annotation_index"][4]
    assert pd.isna(ans["detection_index"][3])
    assert pd.isna(ans["match_value"][3])


def test_reduction_three_true_positives_one_annotation():
    """
    Test matching reduction with three true positives per annotation.
    """
    # arrange
    reduction = MatchingReduction()
    matching = get_test_data_three_true_positives_one_annotation()
    # act
    ans = reduction.reduce_to_exclusive(matching=matching)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 3, "wrong number of rows"
    assert ans["confusion"].equals(pd.Series(data=["tp", "fp", "fp"]))
    assert ans["annotation_index"][0] == matching["annotation_index"][1]
    assert ans["detection_index"][0] == matching["detection_index"][1]


def test_reduction_three_true_positives_one_detection():
    """
    Test matching reduction with three true positives per detection.
    """
    # arrange
    reduction = MatchingReduction()
    matching = get_test_data_three_true_positives_one_detection()
    # act
    ans = reduction.reduce_to_exclusive(matching=matching)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 3, "wrong number of rows"
    assert ans["confusion"].equals(pd.Series(data=["tp", "fn", "fn"]))
    assert ans["annotation_index"][0] == matching["annotation_index"][1]
    assert ans["detection_index"][0] == matching["detection_index"][1]
