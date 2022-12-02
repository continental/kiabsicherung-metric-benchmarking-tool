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
Test resources for the filtering related tests.

"""

import pytest
import pandas as pd
import tempfile
import json

from kia_mbt.kia_filter.kia_filter import KiaFilter
from kia_mbt.config_loader import FilterConfig


def get_test_data():
    """
    Create 3 pandas data frames as test data.

    Returns:
        (pd.DataFrame): ground-truth annotations
        (pd.DataFrame): predictions based on the ground truth data
        (pd.DataFrame): matching of ground-truth and prediction data

    """
    annotation_data = pd.DataFrame(
        data={
            'sample_name': ['mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                            'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                            'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0001'],
            'instance_id': [1000, 1001, 1002, 1003, 2000, 2001],
            'object_id': [1000, 1001, 1002, 1003, 2000, 2001],
            'center': [[1000, 1000], [500, 500], [5, 5], [1000, 1000], [5, 5], [1000, 1000]],
            'size': [[100, 100], [50, 50], [10, 10], [10, 10], [10, 10], [100, 100]],
            'class_id': ['human', 'human', 'human', 'vehicle', 'vehicle', 'human']},
        index=['mv/arb-camera001-0076-cbfa-0000/1000', 'mv/arb-camera001-0076-cbfa-0000/1001',
               'mv/arb-camera001-0076-cbfa-0000/1002', 'mv/arb-camera001-0076-cbfa-0000/1003',
               'mv/arb-camera001-0076-cbfa-0000/1004', 'mv/arb-camera001-0076-cbfa-0001/2000']
    )
    prediction_data = pd.DataFrame(
        data={
            'sample_name': ['mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                            'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                            'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000'],
            'instance_id': [0, 1, 2, 3, 4, 5],
            'object_id': [0, 1, 2, 3, 4, 5],
            'center': [[1000, 1000], [990, 990], [980, 980], [5, 5], [1000, 1000], [1500, 1500]],
            'size': [[100, 100], [100, 100], [100, 100], [10, 10], [10, 10], [10, 10]],
            'class_id': ['human', 'human', 'human', 'human', 'vehicle', 'human'],
            'confidence': [0.8, 0.7, 0.9, 0.8, 0.8, 0.8]},
        index=['mv/arb-camera001-0076-cbfa-0000/0', 'mv/arb-camera001-0076-cbfa-0000/1',
               'mv/arb-camera001-0076-cbfa-0000/2', 'mv/arb-camera001-0076-cbfa-0000/3',
               'mv/arb-camera001-0076-cbfa-0000/4', 'mv/arb-camera001-0076-cbfa-0000/5']
    )
    matching_data = pd.DataFrame(data={
        'annotation_index': ['mv/arb-camera001-0076-cbfa-0000/1003', 'mv/arb-camera001-0076-cbfa-0000/1004',
                             'mv/arb-camera001-0076-cbfa-0000/1000', 'mv/arb-camera001-0076-cbfa-0000/1000',
                             'mv/arb-camera001-0076-cbfa-0000/1000', 'mv/arb-camera001-0076-cbfa-0000/1002',
                             'nan', 'mv/arb-camera001-0076-cbfa-0000/1001', 'mv/arb-camera001-0076-cbfa-0001/2000'],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0000/4', 'None',
                            'mv/arb-camera001-0076-cbfa-0000/0', 'mv/arb-camera001-0076-cbfa-0000/1',
                            'mv/arb-camera001-0076-cbfa-0000/2', 'mv/arb-camera001-0076-cbfa-0000/3',
                            'mv/arb-camera001-0076-cbfa-0000/3', 'None', 'None'],
        'confusion': ['fp', 'fn', 'tp', 'tp', 'tp', 'tp', 'fp', 'fn', 'fn'],
        'class_id': ['vehicle', 'vehicle', 'human', 'human', 'human', 'human', 'human', 'human', 'human'],
        'match_value': [1.0, float('nan'), 1.0, 0.68, 0.47, 1.0, float('nan'), float('nan'), float('nan')],
        'confidence': [0.8, float('nan'), 0.8, 0.7, 0.9, 0.8, 0.8, float('nan'), float('nan')]
    })
    return annotation_data, prediction_data, matching_data


def get_filter_dict():
    """
    Create a set of filters for annotation-, prediction- and matching tables.

    Returns:
        (dict): dictionary with multiple filters for testing

    """
    # filter have the form:
    # <filter_name>: [<column>, <operator>, <value>]
    filter_dict = {
        'annotation_filter': {
            'oid <= 1003': ['object_id', '<=', 1003],
            'cid == human': ['class_id', '==', 'human'],
            'oid != 1004': ['object_id', '!=', 1004],
            'cid in list': ['class_id', 'in', ['cat', 'human', 'dog']],
            'cid not_in list': ['class_id', 'not_in', ['vehicle', 'bicycle']]
        },
        'prediction_filter': {
            'confidence > 0.7': ['confidence', '>', 0.7],
            'confidence <= 0.9': ['confidence', '<=', 0.9],
        },
        'matching_filter': {
            'IoU >= 0.5': ['match_value', '>=', 0.5],
        }
    }
    return filter_dict


@pytest.fixture
def kia_filter_wo_config():
    """
    Yields a KiaFilter object without settings from a config file for testing.

    Returns:
        (KiaFilter): KiaFilter object

    """
    annotation_data, prediction_data, matching_data = get_test_data()
    kia_filter = KiaFilter(annotation_data=annotation_data,
                           prediction_data=prediction_data,
                           matching_data=matching_data,
                           config=FilterConfig())
    yield kia_filter


@pytest.fixture
def kia_filter_with_config():
    """
    Yields a KiaFilter object with a filter config for testing.

    Returns:
        (KiaFilter): KiaFilter object

    """
    annotation_data, prediction_data, matching_data = get_test_data()
    filter_dict = get_filter_dict()
    filter_cfg = FilterConfig(annotation_filter=filter_dict["annotation_filter"],
                              prediction_filter=filter_dict["prediction_filter"],
                              matching_filter=filter_dict["matching_filter"])
    print(filter_cfg)

    kia_filter = KiaFilter(annotation_data=annotation_data,
                           prediction_data=prediction_data,
                           matching_data=matching_data,
                           config=filter_cfg)
    yield kia_filter
