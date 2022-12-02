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
Unit test for mAP score.

"""

import pandas as pd
import numpy as np
from pytest import approx

from kia_mbt.kia_metrics.voc_map import VocMAP
from tests.kia_metrics.conftest import get_empty_data, get_test_data


def test_map_init():
    """
    Test mAP processor initializer.
    """
    # arrange
    map_processor = VocMAP()

    # assert
    assert map_processor.identifier == 1003
    assert map_processor.name == 'VOC mAP'
    assert map_processor.calculate_per_sample is True


################
# VOC mAP 2007 #
################

def test_voc2007_map_empty_data():
    """
    Test computation of VOC mAP with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=False,
                                    ap_integration_mode='11point')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert np.isnan(ans["mAP"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc2007_map_per_class_empty_data():
    """
    Test computation of VOC mAP per class with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=True,
                                    ap_integration_mode='11point')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert np.isnan(ans["mAP"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc_map2007_per_sample_empty_data():
    """
    Test computation of VOC mAP per sample with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=False,
                                        ap_integration_mode='11point')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc_map2007_per_class_per_sample_empty_data():
    """
    Test computation of VOC mAP 2007 per class per sample with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=True,
                                        ap_integration_mode='11point')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc_map2007_fixture_data():
    """
    Test computation of VOC mAP 2007 with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    map_res = 17. / 55.
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=False,
                                    ap_integration_mode='11point')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"
    assert ans["mAP"][0] == approx(map_res), "wrong result"


def test_voc_map_2007_per_class_fixture_data():
    """
    Test computation of VOC mAP 2007 per class with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    map_score = 17. / 55.
    ap_human = 34. / 55.
    ap_vehicle = 0.
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=True,
                                    ap_integration_mode='11point')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 3, "wrong number of columns"
    assert ans["mAP"][0] == approx(map_score)
    assert ans["human"][0] == approx(ap_human)
    assert ans["vehicle"][0] == approx(ap_vehicle)


def test_voc_map_2007_per_sample_fixture_data():
    """
    Test computation of VOC mAP 2007 per sample with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=False,
                                        ap_integration_mode='11point',
                                        eps=np.finfo(float).eps)  # numeric stabilization of corner case
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 1
    assert ans["mAP"][0] == approx(43. / 110.)
    assert np.isnan(ans["mAP"][1])


def test_voc_map_2007_per_class_per_sample_fixture_data():
    """
    Test computation of VOC mAP 2007 per class per sample with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=True,
                                        ap_integration_mode='11point',
                                        eps=np.finfo(float).eps)  # numeric stabilization of corner case
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 3
    assert ans["mAP"][0] == approx(43. / 110.)
    assert np.isnan(ans["mAP"][1])
    assert ans["human"][0] == approx(43. / 55.)
    assert np.isnan(ans["human"][1])
    assert ans["vehicle"][0] == approx(0.0)
    assert np.isnan(ans["vehicle"][1])


################
# VOC mAP 2012 #
################

def test_voc2012_map_empty_data():
    """
    Test computation of VOC mAP with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=False,
                                    ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert np.isnan(ans["mAP"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc2012_map_per_class_empty_data():
    """
    Test computation of VOC mAP 2012 per class with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=True,
                                    ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert np.isnan(ans["mAP"][0]), "wrong result"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc_map2012_per_sample_empty_data():
    """
    Test computation of VOC mAP 2012 per sample with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=False,
                                        ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc_map2012_per_class_per_sample_empty_data():
    """
    Test computation of VOC mAP 2012 per class per sample with empty input.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_empty_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=True,
                                        ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 0, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"


def test_voc_map2012_fixture_data():
    """
    Test computation of VOC mAP 2012 with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    map_res = 19. / 60.
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=False,
                                    ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 1, "wrong number of columns"
    assert ans["mAP"][0] == approx(map_res), "wrong result"


def test_voc_map_2012_per_class_fixture_data():
    """
    Test computation of VOC mAP 2012 per class with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    map_score = 19. / 60.
    ap_human = 19. / 30.
    ap_vehicle = 0.
    # act
    ans = map_processor.calc_global(annotation_data=annotation_data,
                                    prediction_data=prediction_data,
                                    matching=matching,
                                    calculate_per_class=True,
                                    ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 1, "wrong number of rows"
    assert len(ans.columns) == 3, "wrong number of columns"
    assert ans["mAP"][0] == approx(map_score)
    assert ans["human"][0] == approx(ap_human)
    assert ans["vehicle"][0] == approx(ap_vehicle)


def test_voc_map_2012_per_sample_fixture_data():
    """
    Test computation of VOC mAP 2012 per sample with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=False,
                                        ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 1
    assert ans["mAP"][0] == approx(19. / 50.)
    assert np.isnan(ans["mAP"][1])


def test_voc_map_2012_per_class_per_sample_fixture_data():
    """
    Test computation of VOC mAP 2012 per class per sample with default arguments.
    """
    # arrange
    map_processor = VocMAP()
    annotation_data, prediction_data, matching = get_test_data()
    # act
    ans = map_processor.calc_per_sample(annotation_data=annotation_data,
                                        prediction_data=prediction_data,
                                        matching=matching,
                                        calculate_per_class=True,
                                        ap_integration_mode='exact')
    # assert
    assert isinstance(ans, pd.DataFrame), "wrong return type"
    assert len(ans) == 2, "wrong number of rows"
    assert len(ans.columns) == 3
    assert ans["mAP"][0] == approx(19. / 50.)
    assert np.isnan(ans["mAP"][1])
    assert ans["human"][0] == approx(19. / 25)
    assert np.isnan(ans["human"][1])
    assert ans["vehicle"][0] == approx(0.0)
    assert np.isnan(ans["vehicle"][1])
