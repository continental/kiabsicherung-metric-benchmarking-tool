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
Test resources for the metric related tests.

"""

import pandas as pd


def get_empty_data():
    """
    Create empty pandas data frames with columns as in actual data.

    Returns:
        (pd.DataFrame): Empty ground truth annotations.
        (pd.DataFrame): Empty predictions.
        (pd.DataFrame): Empty matching of ground truth and prediction data.
    """
    annotation_cols = ['sample_name', 'instance_id', 'object_id', 'center', 'size', 'class_id']
    annotation_data = pd.DataFrame(data=None, columns=annotation_cols)
    prediction_cols = ['sample_name', 'instance_id', 'object_id', 'center', 'size', 'class_id', 'confidence']
    prediction_data = pd.DataFrame(data=None, columns=prediction_cols)
    matching_cols = ['sample_name', 'annotation_index', 'detection_index', 'confusion', 'class_id', 'match_value', 'confidence']
    matching = pd.DataFrame(data=None, columns=matching_cols)
    return annotation_data, prediction_data, matching


def get_test_data():
    """
    Create 3 pandas data frames as test data.

    Returns:
        (pd.DataFrame): Ground truth annotations.
        (pd.DataFrame): Predictions based on the ground truth data.
        (pd.DataFrame): Matching of ground truth and prediction data.
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
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                        'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                        'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000',
                        'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0000', 'mv/arb-camera001-0076-cbfa-0001'],
        'annotation_index': [None, 'mv/arb-camera001-0076-cbfa-0000/1004',
                             'mv/arb-camera001-0076-cbfa-0000/1000', 'mv/arb-camera001-0076-cbfa-0000/1000',
                             'mv/arb-camera001-0076-cbfa-0000/1000', 'mv/arb-camera001-0076-cbfa-0000/1002',
                             None, 'mv/arb-camera001-0076-cbfa-0000/1001', 'mv/arb-camera001-0076-cbfa-0001/2000'],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0000/4', None,
                            'mv/arb-camera001-0076-cbfa-0000/0', 'mv/arb-camera001-0076-cbfa-0000/1',
                            'mv/arb-camera001-0076-cbfa-0000/2', 'mv/arb-camera001-0076-cbfa-0000/3',
                            'mv/arb-camera001-0076-cbfa-0000/3', None, None],
        'confusion': ['fp', 'fn', 'tp', 'tp', 'tp', 'tp', 'fp', 'fn', 'fn'],
        'class_id': ['vehicle', 'vehicle', 'human', 'human', 'human', 'human', 'human', 'human', 'human'],
        'match_value': [float('nan'), float('nan'), 1.0, 0.68, 0.47, 1.0, float('nan'), float('nan'), float('nan')],
        'confidence': [0.8, float('nan'), 0.8, 0.7, 0.9, 0.8, 0.8, float('nan'), float('nan')]
    })
    return annotation_data, prediction_data, matching
