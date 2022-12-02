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
Test for objects related to filtering. Tested objects are from the files:

"""

import pandas as pd


def test_mbt_filter_init(kia_filter_wo_config):
    """
    Test for the __init__ method of the KiaFilter class.

    Parameters
    ----------
        kia_filter_wo_config (pytest.fixture): KiaFilter object

    """
    assert isinstance(kia_filter_wo_config.annotation_data, pd.DataFrame), 'annotation_data is no pandas.DataFrame'
    assert isinstance(kia_filter_wo_config.prediction_data, pd.DataFrame), 'prediction_data is no pandas.DataFrame'
    assert isinstance(kia_filter_wo_config.matching_data, pd.DataFrame), 'correlation_data is no pandas.DataFrame'


def test_filter_settings(kia_filter_with_config):
    """
    Test if filters are correct initialized if loaded from a config file.

    Parameters
    ----------
        Kia_filter_with_config (pytest.fixture): KiaFilter object

    """
    assert len(kia_filter_with_config.annotation_filter) == 5, 'wrong number of annotation filters'
    assert list(kia_filter_with_config.annotation_filter[0].keys()) == ['filter_info', 'filter_list']
    assert set(x['filter_info'] for x in kia_filter_with_config.annotation_filter) == \
           {'oid <= 1003', 'cid == human', 'oid != 1004', 'cid in list', 'cid not_in list'}

    assert len(kia_filter_with_config.prediction_filter) == 2, 'wrong number of prediction filters'
    assert list(kia_filter_with_config.prediction_filter[0].keys()) == ['filter_info', 'filter_list']
    assert set(x['filter_info'] for x in kia_filter_with_config.prediction_filter) == \
           {'confidence > 0.7', 'confidence <= 0.9'}

    assert len(kia_filter_with_config.matching_filter) == 1, 'wrong number of correlation filters'
    assert list(kia_filter_with_config.matching_filter[0].keys()) == ['filter_info', 'filter_list']
    assert kia_filter_with_config.matching_filter[0]['filter_info'] == 'IoU >= 0.5'


def test_apply_annotation_filter(kia_filter_wo_config):
    """
    Test if a custom filter is correctly applied to annotation data.

    Args:
        kia_filter_wo_config (pytest.fixture): MbtFilter object

    """
    annotation_data = kia_filter_wo_config.annotation_data
    custom_filter = []
    for row_idx in range(annotation_data.shape[0]):
        if annotation_data['size'][row_idx][0] > 33 and annotation_data['size'][row_idx][1] > 33 and \
                (annotation_data['center'][row_idx][0] - 0.5 * annotation_data['size'][row_idx][0] >= 0):
            custom_filter.append(True)
        else:
            custom_filter.append(False)
    kia_filter_wo_config.apply_annotation_filter(info='custom_filter', filter_list=custom_filter)

    assert len(kia_filter_wo_config.annotation_filter) == 1
    assert list(kia_filter_wo_config.annotation_filter[0].keys()) == ['filter_info', 'filter_list']
    assert kia_filter_wo_config.annotation_filter[0]['filter_info'] == 'custom_filter'
    assert kia_filter_wo_config.annotation_filter[0]['filter_list'] == custom_filter


def test_method_get_view(kia_filter_with_config):
    """
    Test if the get_view method works properly. The get_view method creates a pandas DataFrame
    of the remaining matching data considering filters applied on the ground truth-, prediction-,
    and matching table.

    Args:
        kia_filter_with_config (pytest.fixture): KiaFilter object

    """
    view = kia_filter_with_config.get_view()
    assert view.shape[0] == 2
    assert (view['annotation_index'].iloc[0], view['detection_index'].iloc[0]) == \
           ('mv/arb-camera001-0076-cbfa-0000/1000', 'mv/arb-camera001-0076-cbfa-0000/0')
    assert (view['annotation_index'].iloc[1], view['detection_index'].iloc[1]) == \
           ('mv/arb-camera001-0076-cbfa-0000/1002', 'mv/arb-camera001-0076-cbfa-0000/3')
