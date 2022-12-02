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
Unit test number_of_true_positives.

"""

import pandas as pd

from kia_mbt.kia_metrics.number_of_true_positives import NumberOfTruePositives
from tests.kia_metrics.conftest import get_empty_data, get_test_data


def test_number_of_true_positives_init():
    """
    Test number of true positives initialization.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()

    # assert
    assert num_tp_processor.identifier == 1029
    assert num_tp_processor.name == 'Number of True Positives'
    assert num_tp_processor.calculate_per_sample is True


def test_number_of_true_positives_empty_data():
    """
    Test computation of true positives with empty input.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = num_tp_processor.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == 0, "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_number_of_true_positives_per_class_empty_data():
    """
    Test computation of true positives per class with empty input.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = num_tp_processor.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == 0, "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_number_of_true_positives_per_class_per_sample_empty_data():
    """
    Test computation of true positives per class per sample with empty input.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = num_tp_processor.calc_per_sample(annotation_data=annotation_data,
                                           prediction_data=prediction_data,
                                           matching=matching,
                                           calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_number_of_true_positives_fixture_data():
    """
    Test computation of total number of true positives.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = num_tp_processor.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == 4, "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_number_of_true_positives_per_class_fixture_data():
    """
    Test computation of true positives per class with default arguments.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = num_tp_processor.calc_global(annotation_data=annotation_data,
                                       prediction_data=prediction_data,
                                       matching=matching,
                                       calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == 4, "wrong result"
    assert ans["human"][0] == 4, "wrong result for human"
    assert ans["vehicle"][0] == 0, "wrong result for vehicle"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 3, "wrong number of columns"


def test_number_of_true_positives_per_sample_fixture_data():
    """
    Test computation of true positives per sample with default arguments.
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_test_data()
    sample_names = list(annotation_data["sample_name"].unique())
    # act
    ans = num_tp_processor.calc_per_sample(annotation_data=annotation_data,
                                           prediction_data=prediction_data,
                                           matching=matching,
                                           calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == 4, "wrong result for sample 0"
    assert ans["total"][1] == 0, "wrong result for sample 1"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"
    assert [sample_names[i] == ans.index[i] for i in range(len(sample_names))]


def test_number_of_true_positives_per_class_per_sample_fixture_data():
    """
    Test computation of true positives per class per sample with default arguments
    """
    # arrange
    num_tp_processor = NumberOfTruePositives()
    annotation_data, prediction_data, matching = get_test_data()
    sample_names = list(annotation_data["sample_name"].unique())
    # act
    ans = num_tp_processor.calc_per_sample(annotation_data=annotation_data,
                                           prediction_data=prediction_data,
                                           matching=matching,
                                           calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == 4, "wrong result for sample 0 total"
    assert ans["total"][1] == 0, "wrong result for sample 1 total"
    assert ans["human"][0] == 4, "wrong result for sample 0 class human"
    assert ans["vehicle"][0] == 0, "wrong result for sample 0 class vehicle"
    assert ans["human"][1] == 0, "wrong result for sample 1 class human"
    assert pd.isna(ans["vehicle"][1]), "wrong result for sample 1 class vehicle"
    assert [sample_names[i] == ans.index[i] for i in range(len(sample_names))]
