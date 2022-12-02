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
Unit test mean_intersection_over_union.

"""

import pandas as pd
from pytest import approx

from kia_mbt.kia_metrics.mean_intersection_over_union import MeanIntersectionOverUnion
from tests.kia_metrics.conftest import get_empty_data, get_test_data


def test_mean_intersection_over_union_init():
    """
    Test mean intersection over union initialization.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()

    # assert
    assert mean_iou_processor.identifier == 1000
    assert mean_iou_processor.name == 'Mean Intersection Over Union'
    assert mean_iou_processor.calculate_per_sample is True


def test_mean_intersection_over_union_empty_data():
    """
    Test computation of mean IOU with empty input.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = mean_iou_processor.calc_global(annotation_data=annotation_data,
                                         prediction_data=prediction_data,
                                         matching=matching,
                                         calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert pd.isna(ans["total"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_mean_intersection_over_union_per_class_empty_data():
    """
    Test computation of mean IOU per class with empty input.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = mean_iou_processor.calc_global(annotation_data=annotation_data,
                                         prediction_data=prediction_data,
                                         matching=matching,
                                         calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert pd.isna(ans["total"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_mean_intersection_over_union_per_class_per_sample_empty_data():
    """
    Test computation of mean IOU per class per sample with empty input.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = mean_iou_processor.calc_per_sample(annotation_data=annotation_data,
                                             prediction_data=prediction_data,
                                             matching=matching,
                                             calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_mean_intersection_over_union_fixture_data():
    """
    Test computation of mean IOU.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = mean_iou_processor.calc_global(annotation_data=annotation_data,
                                         prediction_data=prediction_data,
                                         matching=matching,
                                         calculate_per_class=False)
    # assert
    res = (1.0 + 0.68 + 0.47 + 1.0 + 0.0 + 0.0 + 0.0) / 7.0
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(res), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_mean_intersection_over_union_per_class_fixture_data():
    """
    Test computation of mean IOU per class with default arguments.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = mean_iou_processor.calc_global(annotation_data=annotation_data,
                                         prediction_data=prediction_data,
                                         matching=matching,
                                         calculate_per_class=True)
    # assert
    res_total = (1.0 + 0.68 + 0.47 + 1.0 + 0.0 + 0.0 + 0.0) / 7.0
    res_human = (1.0 + 0.68 + 0.47 + 1.0 + 0.0 + 0.0) / 6.0
    res_vehicle = 0.0

    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(res_total), "wrong result"
    assert ans["human"][0] == approx(res_human), "wrong result for human"
    assert ans["vehicle"][0] == approx(res_vehicle), "wrong result for vehicle"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 3, "wrong number of columns"


def test_mean_intersection_over_union_per_sample_fixture_data():
    """
    Test computation of mean IOU per sample with default arguments.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_test_data()
    sample_names = list(annotation_data["sample_name"].unique())
    # act
    ans = mean_iou_processor.calc_per_sample(annotation_data=annotation_data,
                                             prediction_data=prediction_data,
                                             matching=matching,
                                             calculate_per_class=False)
    # assert
    res_total_0 = (1.0 + 0.68 + 0.47 + 1.0 + 0.0 + 0.0) / 6.0
    res_total_1 = 0.0

    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(res_total_0), "wrong result for sample 0"
    assert ans["total"][1] == approx(res_total_1), "wrong result for sample 1"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"
    assert [sample_names[i] == ans.index[i] for i in range(len(sample_names))]


def test_mean_intersection_over_union_per_class_per_sample_fixture_data():
    """
    Test computation of mean IOU per class per sample with default arguments.
    """
    # arrange
    mean_iou_processor = MeanIntersectionOverUnion()
    annotation_data, prediction_data, matching = get_test_data()
    sample_names = list(annotation_data["sample_name"].unique())
    # act
    ans = mean_iou_processor.calc_per_sample(annotation_data=annotation_data,
                                             prediction_data=prediction_data,
                                             matching=matching,
                                             calculate_per_class=True)
    # assert
    res_total_0 = (1.0 + 0.68 + 0.47 + 1.0 + 0.0 + 0.0) / 6.0
    res_total_1 = 0.0
    res_human_0 = (1.0 + 0.68 + 0.47 + 1.0 + 0.0) / 5.0
    res_human_1 = 0.0
    res_vehicle_0 = 0.0
    # res_vehilce_1 = float('nan')
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(res_total_0), "wrong result for sample 0 total"
    assert ans["total"][1] == approx(res_total_1), "wrong result for sample 1 total"
    assert ans["human"][0] == approx(res_human_0), "wrong result for sample 0 class human"
    assert ans["vehicle"][0] == approx(res_vehicle_0), "wrong result for sample 0 class vehicle"
    assert ans["human"][1] == approx(res_human_1), "wrong result for sample 1 class human"
    assert pd.isna(ans["vehicle"][1]), "wrong result for sample 1 class vehicle"
    assert [sample_names[i] == ans.index[i] for i in range(len(sample_names))]
