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
Unit test precision-recall curves.

"""

import pandas as pd
import numpy as np

from kia_mbt.kia_metrics.precision_recall_curve import PrecisionRecallCurve
from tests.kia_metrics.conftest import get_empty_data, get_test_data


def test_precision_recall_curve_init():
    """
    Test precision recall curve initializer.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()

    # assert
    assert precision_recall.identifier == 1040
    assert precision_recall.name == 'Precision-Recall Curve'
    assert precision_recall.calculate_per_sample is True


def test_precision_recall_empty_data():
    """
    Test computation of precision-recall with empty input.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = precision_recall.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert np.isnan(ans["total"][0][0])
    assert np.isnan(ans["total"][0][1])
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_recall_per_class_empty_data():
    """
    Test computation of precision-recall per class with empty input.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = precision_recall.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert np.isnan(ans["total"][0][0])
    assert np.isnan(ans["total"][0][1])
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_recall_per_class_per_sample_empty_data():
    """
    Test computation of precision-recall per class per sample with empty input.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = precision_recall.calc_per_sample(annotation_data=annotation_data,
                                           prediction_data=prediction_data,
                                           matching=matching,
                                           calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_recall_fixture_data():
    """
    Test computation of precision-recall with default arguments.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_test_data()

    recall_cmp = np.asarray([1/7, 1/7, 2/7, 3/7, 3/7, 4/7])
    precision_cmp = np.asarray([1.0, 1/2, 2/3, 3/4, 3/5, 4/6])
    # act
    ans = precision_recall.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"
    recall = ans["total"][0][0]
    precision = ans["total"][0][1]
    np.testing.assert_almost_equal(recall, recall_cmp)
    np.testing.assert_almost_equal(precision, precision_cmp)


def test_precision_recall_per_class_fixture_data():
    """
    Test computation of precision-recall per class with default arguments.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_test_data()

    recall_total = np.asarray([1/7, 1/7, 2/7, 3/7, 3/7, 4/7])
    precision_total = np.asarray([1.0, 1/2, 2/3, 3/4, 3/5, 4/6])
    recall_ped = np.asarray([1/6, 2/6, 3/6, 3/6, 4/6])
    precision_ped = np.asarray([1.0, 1.0, 1.0, 3/4, 4/5])
    recall_veh = np.asarray([0.0])
    precision_veh = np.asarray([0.0])
    # act
    ans = precision_recall.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 3, "wrong number of columns"
    recall_ans_tot = ans["total"][0][0]
    np.testing.assert_almost_equal(recall_ans_tot, recall_total)
    precision_ans_tot = ans["total"][0][1]
    np.testing.assert_almost_equal(precision_ans_tot, precision_total)
    recall_ans_ped = ans["human"][0][0]
    np.testing.assert_almost_equal(recall_ans_ped, recall_ped)
    precision_ans_ped = ans["human"][0][1]
    np.testing.assert_almost_equal(precision_ans_ped, precision_ped)
    recall_ans_veh = ans["vehicle"][0][0]
    np.testing.assert_almost_equal(recall_ans_veh, recall_veh)
    precision_ans_veh = ans["vehicle"][0][1]
    np.testing.assert_almost_equal(precision_ans_veh, precision_veh)


def test_precsion_recall_per_sample_fixture_data():
    """
    Test computation of precision-recall per sample with default arguments.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_test_data()

    recall_sample1 = np.asarray([1/6, 1/6, 2/6, 3/6, 3/6, 4/6])
    precision_sample1 = np.asarray([1.0, 1/2, 2/3, 3/4, 3/5, 4/6])

    recall_sample2 = np.asarray([0.0])
    # precision_sample2 = np.asarray([np.nan])
    # act
    ans = precision_recall.calc_per_sample(annotation_data=annotation_data,
                                           prediction_data=prediction_data,
                                           matching=matching,
                                           calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 1
    recall_ans_sample1 = ans["total"][0][0]
    np.testing.assert_almost_equal(recall_ans_sample1, recall_sample1)
    precision_ans_sample1 = ans["total"][0][1]
    np.testing.assert_almost_equal(precision_ans_sample1, precision_sample1)
    recall_ans_sample2 = ans["total"][1][0]
    np.testing.assert_almost_equal(recall_ans_sample2, recall_sample2)
    precision_ans_sample2 = ans["total"][1][1]
    assert np.isnan(precision_ans_sample2)


def test_precision_recall_per_class_per_sample_fixture_data():
    """
    Test computation of precision-recall per class per sample with default arguments.
    """
    # arrange
    precision_recall = PrecisionRecallCurve()
    annotation_data, prediction_data, matching = get_test_data()

    recall_sample1 = np.asarray([1/6, 1/6, 2/6, 3/6, 3/6, 4/6])
    recall_ped_sample1 = np.asarray([1/5, 2/5, 3/5, 3/5, 4/5])
    recall_veh_sample1 = np.asarray([0.0])

    precision_sample1 = np.asarray([1.0, 1/2, 2/3, 3/4, 3/5, 4/6])
    precision_ped_sample1 = np.asarray([1.0, 1.0, 1.0, 3/4, 4/5])
    precision_veh_sample1 = np.asarray([0.0])

    recall_sample2 = np.asarray([0.0])
    recall_ped_sample2 = np.asarray([0.0])
    # recall_veh_sample2 = np.asarray([np.nan])

    # precision_sample2 = np.asarray([np.nan])
    # precision_ped_sample2 = np.asarray([np.nan])
    # precision_veh_sample2 = np.asarray([np.nan])

    # act
    ans = precision_recall.calc_per_sample(annotation_data=annotation_data,
                                           prediction_data=prediction_data,
                                           matching=matching,
                                           calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame)
    assert len(ans) == 2
    assert len(ans.columns) == 3

    # sample 1
    ans_recall_sample1 = ans["total"][0][0]
    np.testing.assert_almost_equal(recall_sample1, ans_recall_sample1)
    ans_precision_sample1 = ans["total"][0][1]
    np.testing.assert_almost_equal(precision_sample1, ans_precision_sample1)
    ans_recall_ped_sample1 = ans["human"][0][0]
    np.testing.assert_almost_equal(recall_ped_sample1, ans_recall_ped_sample1)
    ans_precision_ped_sample1 = ans["human"][0][1]
    np.testing.assert_almost_equal(precision_ped_sample1, ans_precision_ped_sample1)
    ans_recall_veh_sample1 = ans["vehicle"][0][0]
    np.testing.assert_almost_equal(recall_veh_sample1, ans_recall_veh_sample1)
    ans_precision_veh_sample1 = ans["vehicle"][0][1]
    np.testing.assert_almost_equal(precision_veh_sample1, ans_precision_veh_sample1)
    # sample 2
    ans_recall_sample2 = ans["total"][1][0]
    np.testing.assert_almost_equal(recall_sample2, ans_recall_sample2)
    ans_precision_sample2 = ans["total"][1][1]
    assert np.isnan(ans_precision_sample2)
    ans_recall_ped_sample2 = ans["human"][1][0]
    np.testing.assert_almost_equal(recall_ped_sample2, ans_recall_ped_sample2)
    ans_precision_ped_sample2 = ans["human"][1][1]
    assert np.isnan(ans_precision_ped_sample2)
    ans_prec_recall_veh_sample2 = ans["vehicle"][1]
    assert np.isnan(ans_prec_recall_veh_sample2)
