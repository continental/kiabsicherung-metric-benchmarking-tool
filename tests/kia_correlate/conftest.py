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
Test resources for the correlation related tests.

"""

import pandas as pd


def get_empty_data():
    """
    Create empty pandas data frames with columns as in actual data.

    Returns
    -------
        (pd.DataFrame): Empty ground truth annotations.
        (pd.DataFrame): Empty predictions.
        (pd.DataFrame): Empty matching of ground truth and prediction data.

    """
    annotation_cols = ['sample_name', 'instance_id', 'object_id', 'center', 'size', 'class_id']
    annotation_data = pd.DataFrame(data=None, columns=annotation_cols)
    prediction_cols = ['sample_name', 'instance_id', 'object_id', 'center', 'size', 'class_id', 'confidence']
    prediction_data = pd.DataFrame(data=None, columns=prediction_cols)
    matching_cols = ['sample_name', 'annotation_index', 'detection_index', 'confusion', 'class_id', 'matching_value', 'confidence']
    matching = pd.DataFrame(data=None, columns=matching_cols)
    return annotation_data, prediction_data, matching


def get_test_data():
    """
    Create 3 pandas data frames as test data.

    Returns
    -------
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


def get_test_data_clipped_boxes():
    """
    Create pandas data frames to test matching for clipped bounding-boxes.

    Returns
    -------
        (pd.DataFrame): Ground truth annotations.
        (pd.DataFrame): Predictions based on the ground truth data.

    """
    annotation_data = pd.DataFrame(
        data={
            'sample_name': ['mv/arb-camera001-0076-cbfa-0000',  # 0
                            ],
            'instance_id': [1000,  # 0
                            ],
            'object_id': [1000,  # 0
                          ],
            'center': [[1920, 100],  # 0
                       ],
            'size': [[400, 200],  # 0
                     ],
            'class_id': ['human',  # 0
                         ]
            },
        index=['mv/arb-camera001-0076-cbfa-0000/1000',  # 0
               ]
    )
    prediction_data = pd.DataFrame(
        data={
            'sample_name': ['mv/arb-camera001-0076-cbfa-0000',  # 0
                            ],
            'instance_id': [0,  # 0
                            ],
            'object_id': [0,  # 0
                          ],
            'center': [[1820, 100],  # 0
                       ],
            'size': [[200, 200],  # 0
                     ],
            'class_id': ['human',  # 0
                         ],
            'confidence': [0.8,  # 0
                           ]
            },
        index=['mv/arb-camera001-0076-cbfa-0000/0',  # 0
               ]
    )
    return annotation_data, prediction_data


def get_test_data_single_tp():
    """
    Create pandas data frame as iou matching reduction test data.

    Returns
    -------
        (pd.DataFrame): Example matching data with single true-positive.

    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001', ],
        'annotation_index': ['mv/arb-camera001-0076-cbfa-0001/1000', ],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10', ],
        'confusion': ['tp', ],
        'class_id': ['human', ],
        'match_value': [0.5, ],
        'confidence': [0.5, ],
    })
    return matching


def get_test_data_single_fp_fn():
    """
    Create pandas data frame as matching threshold test data.

    Returns:
        (pd.DataFrame): Example matching data with single true-positive.
    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001', 'mv/arb-camera001-0076-cbfa-0002', ],
        'annotation_index': [None, 'mv/arb-camera001-0076-cbfa-0001/2000', ],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10', None, ],
        'confusion': ['fp', 'fn', ],
        'class_id': ['human', 'vehicle', ],
        'match_value': [float('nan'), float('nan'), ],
        'confidence': [0.5, float('nan'), ],
    })
    return matching


def get_test_data_tp_with_alternative_matches():
    """
    Create pandas data frame as matching threshold test data.

    Returns:
        (pd.DataFrame): Example matching data with multiple true positives.
    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001',  # 0
                        'mv/arb-camera001-0076-cbfa-0001',  # 1
                        'mv/arb-camera001-0076-cbfa-0001',  # 2
                        'mv/arb-camera001-0076-cbfa-0001',  # 3
                        'mv/arb-camera001-0076-cbfa-0001',  # 4
                        ],
        'annotation_index': ['mv/arb-camera001-0076-cbfa-0001/1000',  # 0
                             'mv/arb-camera001-0076-cbfa-0001/1000',  # 1
                             'mv/arb-camera001-0076-cbfa-0001/1000',  # 2
                             'mv/arb-camera001-0076-cbfa-0001/1001',  # 3
                             'mv/arb-camera001-0076-cbfa-0001/1002',  # 4
                             ],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10',  # 0
                            'mv/arb-camera001-0076-cbfa-0001/11',  # 1
                            'mv/arb-camera001-0076-cbfa-0001/12',  # 2
                            'mv/arb-camera001-0076-cbfa-0001/11',  # 3
                            'mv/arb-camera001-0076-cbfa-0001/10',  # 4
                            ],
        'confusion': ['tp',  # 0
                      'tp',  # 1
                      'tp',  # 2
                      'tp',  # 3
                      'tp',  # 4
                      ],
        'class_id': ['human',  # 0
                     'human',  # 1
                     'human',  # 2
                     'human',  # 3
                     'human',  # 4
                     ],
        'match_value': [1.0,  # 0
                        0.3,  # 1
                        0.5,  # 2
                        1.0,  # 3
                        0.5,  # 4
                        ],
        'confidence': [1.0,  # 0
                       0.3,  # 1
                       0.5,  # 2
                       1.0,  # 3
                       0.5,  # 4
                       ],
    })
    return matching


def get_test_data_three_true_positives_one_annotation():
    """
    Create pandas data frame as matching reduction test data.

    Returns:
        (pd.DataFrame): Example matching data.
    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001',  # 0
                        'mv/arb-camera001-0076-cbfa-0001',  # 1
                        'mv/arb-camera001-0076-cbfa-0001',  # 2
                        ],
        'annotation_index': ['mv/arb-camera001-0076-cbfa-0001/1000',  # 0
                             'mv/arb-camera001-0076-cbfa-0001/1000',  # 1
                             'mv/arb-camera001-0076-cbfa-0001/1000',  # 2
                             ],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10',  # 0
                            'mv/arb-camera001-0076-cbfa-0001/11',  # 1
                            'mv/arb-camera001-0076-cbfa-0001/12',  # 2
                            ],
        'confusion': ['tp',  # 0
                      'tp',  # 1
                      'tp',  # 2
                      ],
        'class_id': ['human',  # 0
                     'human',  # 1
                     'human',  # 2
                     ],
        'match_value': [0.7,  # 0
                        0.9,  # 1
                        0.8,  # 2
                        ],
        'confidence': [0.8,  # 0
                       0.9,  # 1
                       0.7,  # 2
                       ],
    })
    return matching


def get_test_data_three_true_positives_one_detection():
    """
    Create pandas data frame as matching reduction test data.

    Returns:
        (pd.DataFrame): Example matching data.
    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001',  # 0
                        'mv/arb-camera001-0076-cbfa-0001',  # 1
                        'mv/arb-camera001-0076-cbfa-0001',  # 2
                        ],
        'annotation_index': ['mv/arb-camera001-0076-cbfa-0001/1001',  # 0
                             'mv/arb-camera001-0076-cbfa-0001/1002',  # 1
                             'mv/arb-camera001-0076-cbfa-0001/1003',  # 2
                             ],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10',  # 0
                            'mv/arb-camera001-0076-cbfa-0001/10',  # 1
                            'mv/arb-camera001-0076-cbfa-0001/10',  # 2
                            ],
        'confusion': ['tp',  # 0
                      'tp',  # 1
                      'tp',  # 2
                      ],
        'class_id': ['human',  # 0
                     'human',  # 1
                     'human',  # 2
                     ],
        'match_value': [0.7,  # 0
                        0.9,  # 1
                        0.8,  # 2
                        ],
        'confidence': [0.8,  # 0
                       0.8,  # 1
                       0.8,  # 2
                       ],
    })
    return matching


def get_test_data_true_positives_multiple_occurences():
    """
    Create pandas data frame as matching reduction test data.

    Returns:
        (pd.DataFrame): Example matching data.
    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001',  # 0
                        'mv/arb-camera001-0076-cbfa-0001',  # 1
                        'mv/arb-camera001-0076-cbfa-0001',  # 2
                        'mv/arb-camera001-0076-cbfa-0002',  # 3
                        'mv/arb-camera001-0076-cbfa-0002',  # 4
                        'mv/arb-camera001-0076-cbfa-0002',  # 5
                        ],
        'annotation_index': ['mv/arb-camera001-0076-cbfa-0001/1001',  # 0
                             'mv/arb-camera001-0076-cbfa-0001/1001',  # 1
                             'mv/arb-camera001-0076-cbfa-0001/1001',  # 2
                             'mv/arb-camera001-0076-cbfa-0001/1002',  # 3
                             'mv/arb-camera001-0076-cbfa-0001/1003',  # 4
                             'mv/arb-camera001-0076-cbfa-0001/1004',  # 5
                             ],
        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10',  # 0
                            'mv/arb-camera001-0076-cbfa-0001/11',  # 1
                            'mv/arb-camera001-0076-cbfa-0001/12',  # 2
                            'mv/arb-camera001-0076-cbfa-0001/13',  # 3
                            'mv/arb-camera001-0076-cbfa-0001/13',  # 4
                            'mv/arb-camera001-0076-cbfa-0001/13',  # 5
                            ],
        'confusion': ['tp',  # 0
                      'tp',  # 1
                      'tp',  # 2
                      'tp',  # 3
                      'tp',  # 4
                      'tp',  # 5
                      ],
        'class_id': ['human',  # 0
                     'human',  # 1
                     'human',  # 2
                     'human',  # 3
                     'human',  # 4
                     'human',  # 5
                     ],
        'match_value': [0.9,  # 0
                        0.9,  # 1
                        0.9,  # 2
                        0.8,  # 3
                        0.7,  # 4
                        0.6,  # 5
                        ],
        'confidence': [0.9,  # 0
                       0.8,  # 1
                       0.7,  # 2
                       0.9,  # 3
                       0.9,  # 4
                       0.9,  # 5
                       ],
    })
    return matching


def get_test_data_matching():
    """
    Create pandas data frame as iou matching reduction test data.

    Returns:
        (pd.DataFrame): Example matching data.
    """
    matching = pd.DataFrame(data={
        'sample_name': ['mv/arb-camera001-0076-cbfa-0001',  # 0
                        'mv/arb-camera001-0076-cbfa-0002',  # 1
                        'mv/arb-camera001-0076-cbfa-0003',  # 2
                        'mv/arb-camera001-0076-cbfa-0004',  # 3
                        'mv/arb-camera001-0076-cbfa-0004',  # 4
                        'mv/arb-camera001-0076-cbfa-0004',  # 5
                        'mv/arb-camera001-0076-cbfa-0004',  # 6
                        ],

        'annotation_index': ['mv/arb-camera001-0076-cbfa-0001/1000',    # 0
                             None,                                      # 1
                             'mv/arb-camera001-0076-cbfa-0003/3000',    # 2
                             'mv/arb-camera001-0076-cbfa-0004/4000',    # 3
                             'mv/arb-camera001-0076-cbfa-0004/4001',    # 4
                             'mv/arb-camera001-0076-cbfa-0004/4001',    # 5
                             'mv/arb-camera001-0076-cbfa-0004/4002',    # 6
                             ],

        'detection_index': ['mv/arb-camera001-0076-cbfa-0001/10',    # 0
                            'mv/arb-camera001-0076-cbfa-0002/20',    # 1
                            None,                                    # 2
                            'mv/arb-camera001-0076-cbfa-0004/40',    # 3
                            'mv/arb-camera001-0076-cbfa-0004/41',    # 4
                            'mv/arb-camera001-0076-cbfa-0004/42',    # 5
                            'mv/arb-camera001-0076-cbfa-0004/41',    # 6
                            ],

        'confusion': ['tp',  # 0 + stays
                      'fp',  # 1 + stays
                      'fn',  # 2 + stays
                      'tp',  # 3 - remove -> resolved into fn/fp pair
                      'tp',  # 4 + stays (perfect match)
                      'tp',  # 5 - remove -> causes new fp, annotation_index still exists
                      'tp',  # 6 - remove -> causes new fn, detection_index still exists
                      ],

        'class_id': ['human',    # 0
                     'human',    # 1
                     'vehicle',  # 2
                     'human',    # 3
                     'human',    # 4
                     'human',    # 5
                     'human',    # 6
                     ],

        'match_value': [0.8,             # 0 + stays
                        float('nan'),    # 1 + stays
                        float('nan'),    # 2 + stays
                        0.4,             # 3 - remove
                        1.0,             # 4 + stays (perfect match)
                        0.3,             # 5 - remove
                        0.2,             # 6 - remove
                        ],

        'confidence': [0.9,             # 0
                       0.8,             # 1
                       float('nan'),    # 2
                       0.4,             # 3
                       1.0,             # 4
                       0.3,             # 5
                       0.2,             # 6
                       ],
    })
    return matching
