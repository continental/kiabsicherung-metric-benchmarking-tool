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
Resources for output writer unittest.

"""

import pandas as pd


def get_empty_global_metric_data():
    """
    Get empty output for global metric computation.
    """
    global_metrics = list()
    return global_metrics


def get_global_metric_test_data():
    """
    Get example output for global metric computation.
    """
    global_metrics = list()

    pd_recall = pd.DataFrame(data={'total': {0: 0.16}, 'human': {0: 0.21}, 'vehicle': {0: 0.0}})
    recall = [1030, 'Recall', pd_recall]

    pd_num_fp = pd.DataFrame(data={'total': {0: 9115}, 'human': {0: 6744}, 'vehicle': {0: 2371}})
    num_fp = [1031, 'Number of False Negatives', pd_num_fp]

    global_metrics.append(recall)
    global_metrics.append(num_fp)
    return global_metrics


def get_empty_per_sample_test_data():
    """
    Get empty output for per sample metric computation.
    """
    sample_metrics = list()
    return sample_metrics


def get_per_sample_metric_test_data():
    """
    Get example output for per sample metric computation.
    """
    per_sample_metrics = list()

    pd_recall = pd.DataFrame(data={'total': {'mv/arb-camera001-006d842f-0000': 0.26,
                                             'mv/arb-camera001-006d842f-0001': 0.36,
                                             'mv/arb-camera001-006d842f-0002': 0.23,
                                             'mv/arb-camera001-006d842f-0003': 0.23,
                                             'mv/arb-camera001-006d842f-0005': 0.14,
                                             },
                                   'human': {'mv/arb-camera001-006d842f-0000': 0.33,
                                             'mv/arb-camera001-006d842f-0001': 0.50,
                                             'mv/arb-camera001-006d842f-0002': 0.30,
                                             'mv/arb-camera001-006d842f-0003': 0.30,
                                             'mv/arb-camera001-006d842f-0005': 0.18,
                                             },
                                   'vehicle': {'mv/arb-camera001-006d842f-0000': 0.00,
                                               'mv/arb-camera001-006d842f-0001': 0.00,
                                               'mv/arb-camera001-006d842f-0002': 0.00,
                                               'mv/arb-camera001-006d842f-0003': 0.00,
                                               'mv/arb-camera001-006d842f-0005': 0.00,
                                               }})
    recall = [1030, 'Recall', pd_recall]

    pd_num_fp = pd.DataFrame(data={'total': {'mv/arb-camera001-006d842f-0000': 11,
                                             'mv/arb-camera001-006d842f-0001': 7,
                                             'mv/arb-camera001-006d842f-0002': 10,
                                             'mv/arb-camera001-006d842f-0003': 10,
                                             'mv/arb-camera001-006d842f-0005': 12,
                                             },
                                   'human': {'mv/arb-camera001-006d842f-0000': 8,
                                             'mv/arb-camera001-006d842f-0001': 4,
                                             'mv/arb-camera001-006d842f-0002': 7,
                                             'mv/arb-camera001-006d842f-0003': 7,
                                             'mv/arb-camera001-006d842f-0005': 9,
                                             },
                                   'vehicle': {'mv/arb-camera001-006d842f-0000': 3,
                                               'mv/arb-camera001-006d842f-0001': 3,
                                               'mv/arb-camera001-006d842f-0002': 3,
                                               'mv/arb-camera001-006d842f-0003': 3,
                                               'mv/arb-camera001-006d842f-0005': 3,
                                               }})
    num_fp = [1031, 'Number of False Negatives', pd_num_fp]

    per_sample_metrics.append(recall)
    per_sample_metrics.append(num_fp)
    return per_sample_metrics
