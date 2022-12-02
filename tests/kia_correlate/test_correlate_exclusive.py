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
Unit test BoxCorrelator exclusive matching.

"""

from kia_mbt.kia_correlate.box_correlator import BoxCorrelator
from tests.kia_correlate.conftest import get_empty_data
from tests.kia_correlate.conftest import get_test_data


def test_correlate_exclusive_empty_data():
    """
    Test case with empty data.
    """
    # arrange
    box_correlator = BoxCorrelator(threshold=0.5,
                                   matching_type="exclusive")
    annotation_data, prediction_data, _ = get_empty_data()
    # act
    ans = box_correlator(annotation_data=annotation_data,
                         detection_data=prediction_data)
    # assert
    assert ans.empty is True


def test_correlate_exclusive():
    """
    Test case with true positives, false positives and false negatives.
    """
    # arrange
    box_correlator = BoxCorrelator(threshold=0.5,
                                   matching_type="exclusive")
    annotation_data, prediction_data, _ = get_test_data()
    # act
    ans = box_correlator(annotation_data=annotation_data,
                         detection_data=prediction_data)
    confusion_counts = ans["confusion"].value_counts()
    # assert
    assert ans.shape[0] == 9
    assert ans.shape[1] == 7
    assert confusion_counts["tp"] == 3
    assert confusion_counts["fp"] == 3
    assert confusion_counts["fn"] == 3
