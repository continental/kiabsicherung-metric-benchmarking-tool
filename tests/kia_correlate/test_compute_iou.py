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
Test IOU computation.

"""

import pytest
from pytest import approx

from kia_mbt.kia_correlate.box_correlator import BoxCorrelator


def test_compute_iou_full_overlap():
    """
    Test IOU computation for bounding-boxes with full overlap.
    """
    # arrange
    correlator = BoxCorrelator()
    # initialize boxes
    center1 = (960, 540)
    size1 = (100, 100)
    center2 = (960, 540)
    size2 = (100, 100)
    # act
    iou = correlator._compute_iou(center1=center1,
                                  size1=size1,
                                  center2=center2,
                                  size2=size2)
    # assert
    assert iou == approx(1.0, abs=1e-12)


@pytest.mark.parametrize("center2, size2",
                         [((1010, 490), (100, 100)),  # second box to top left
                          ((1010, 590), (100, 100)),  # second box to top right
                          ((910, 490), (100, 100)),  # second box to upper left
                          ((910, 590), (100, 100)),  # second box to upper right
                         ]
)
def test_compute_iou_partial_overlap(center2, size2):
    """
    Test IOU computation for bounding-boxes with partial overlap.
    """
    # arrange
    correlator = BoxCorrelator()
    # reference box
    center1 = (960, 540)  # c_x, c_y
    size1 = (100, 100)  # width, height
    # act
    iou = correlator._compute_iou(center1=center1,
                                  size1=size1,
                                  center2=center2,
                                  size2=size2)
    # assert
    assert iou == approx(1.0 / 7.0, abs=1e-12)


@pytest.mark.parametrize("center2, size2",
                         [((850, 540), (100, 50)),  # second box to the left
                          ((1060, 540), (100, 50)),  # second box to the right
                          ((960, 590), (100, 50)),  # second box "below"
                          ((960, 490), (100, 50)),  # second box "above"
                         ]
)
def test_compute_iou_no_overlap(center2, size2):
    """
    Test IOU computation for bounding-boxes with no overlap.
    """
    # arrange
    correlator = BoxCorrelator()
    # reference box
    center1 = (960, 540) # c_x, c_y
    size1 = (100, 50) # width, height
    # act
    iou = correlator._compute_iou(center1=center1,
                                  size1=size1,
                                  center2=center2,
                                  size2=size2)
    # assert
    assert iou == approx(0.0, abs=1e-12)


@pytest.mark.parametrize("center1, size1, center2, size2, iou_result",
                         [((850, 540), (100, 50), (960, 540), (100, 50), 0.0),
                          ((960, 540), (100, 50), (850, 540), (100, 50), 0.0),
                          ((960, 540), (100, 100), (910, 490), (100, 100), 1.0 / 7.0),
                          ((1920, 10), (40, 20), (1910, 10), (20, 20), 1.0),
                          ((1910, 10), (20, 20), (1920, 10), (40, 20), 1.0),
                          ((0, 0), (20, 20), (5, 5), (5, 5), 0.25),
                          ((5, 5), (5, 5), (0, 0), (20, 20), 0.25),
                         ]
)
def test_compute_iou_with_clipping(center1, size1, center2, size2, iou_result):
    """
    Test IOU computation with bounding-boxes using clipping.
    """
    # arrange
    correlator = BoxCorrelator()
    # reference box
    clip_x = (0.0, 1920.0)
    clip_y = (0.0, 1280.0)
    # act
    iou = correlator._compute_iou(center1=center1,
                                  size1=size1,
                                  center2=center2,
                                  size2=size2,
                                  clip_x=clip_x,
                                  clip_y=clip_y)
    # assert
    assert iou == approx(iou_result)
