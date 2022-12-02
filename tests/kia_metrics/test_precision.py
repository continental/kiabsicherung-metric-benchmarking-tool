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
Unit test precision processor.

"""

import pandas as pd
from pytest import approx

from kia_mbt.kia_metrics.precision import Precision
from tests.kia_metrics.conftest import get_empty_data, get_test_data


def test_precision_init():
    """
    Test precision processor initialization.
    """
    # arrange
    precision_processor = Precision()

    # assert
    assert precision_processor.identifier == 1027
    assert precision_processor.name == 'Precision'
    assert precision_processor.calculate_per_sample is True


def test_precision_empty_data():
    """
    Test computation of precision with empty input.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = precision_processor.calc_global(annotation_data=annotation_data,
                                          prediction_data=prediction_data,
                                          matching=matching,
                                          calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert pd.isna(ans["total"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_per_class_empty_data():
    """
    Test computation of precision per class with empty input.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = precision_processor.calc_global(annotation_data=annotation_data,
                                          prediction_data=prediction_data,
                                          matching=matching,
                                          calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert pd.isna(ans["total"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_per_class_per_sample_empty_data():
    """
    Test computation of precision per class per sample with empty input.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = precision_processor.calc_per_sample(annotation_data=annotation_data,
                                              prediction_data=prediction_data,
                                              matching=matching,
                                              calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_fixture_data():
    """
    Test computation of precision with default arguments.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = precision_processor.calc_global(annotation_data=annotation_data,
                                          prediction_data=prediction_data,
                                          matching=matching,
                                          calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(2.0 / 3.0), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_precision_per_class_fixture_data():
    """
    Test computation of precision per class with default arguments.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = precision_processor.calc_global(annotation_data=annotation_data,
                                          prediction_data=prediction_data,
                                          matching=matching,
                                          calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(4.0 / 6.0), "wrong result"
    assert ans["human"][0] == approx(4.0 / 5.0), "wrong result for class human"
    assert ans["vehicle"][0] == approx(0.0), "wrong result for class vehicle"
    assert len(ans) == 1, "wrong  number of rows"
    assert len(ans.columns) == 3, "wrong number of columns"


def test_precision_per_sample_fixture_data():
    """
    Test computation of precision per sample with default arguments.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_test_data()
    sample_names = list(annotation_data["sample_name"].unique())
    # act
    ans = precision_processor.calc_per_sample(annotation_data=annotation_data,
                                              prediction_data=prediction_data,
                                              matching=matching,
                                              calculate_per_class=False)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(4.0 / 6.0), "wrong result for sample 0"
    assert pd.isna(ans["total"][1]), "wrong result for sample 1"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"
    assert [sample_names[i] == ans.index[i] for i in range(len(sample_names))]


def test_precision_per_class_per_sample_fixture_data():
    """
    Test computation of precision per class per sample with default arguments.
    """
    # arrange
    precision_processor = Precision()
    annotation_data, prediction_data, matching = get_test_data()
    sample_names = list(annotation_data["sample_name"].unique())
    # act
    ans = precision_processor.calc_per_sample(annotation_data=annotation_data,
                                              prediction_data=prediction_data,
                                              matching=matching,
                                              calculate_per_class=True)
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert ans["total"][0] == approx(4.0 / 6.0), "wrong result for sample 0 total"
    assert pd.isna(ans["total"][1]), "wrong result for sample 1 total"
    assert ans["human"][0] == (4.0 / 5.0), "wrong result for sample 0 class human"
    assert ans["vehicle"][0] == approx(0.0), "wrong result for sample 0 class vehicle"
    assert pd.isna(ans["human"][1]), "wrong result for sample 1 class human"
    assert pd.isna(ans["vehicle"][1]), "wrong result for sample 1 class vehicle"
    assert [sample_names[i] == ans.index[i] for i in range(len(sample_names))]
