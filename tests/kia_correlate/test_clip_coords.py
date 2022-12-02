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
Unit test truncate_coordinates.

"""

import pytest
from pytest import approx

from kia_mbt.kia_correlate.box_correlator import BoxCorrelator


@pytest.mark.parametrize(
    "test_input, expected",  # [x_min, y_min, x_max, y_max], [x_min, y_min, x_max, y_max]
    [([0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]),
     ([100.0, 50.0, 200.0, 100.0], [100.0, 50.0, 200.0, 100.0]),
     ([-50.0, 1000.0, 50.0, 1200.0], [-50.0, 1000.0, 50.0, 1200.0]),
     ([1900.0, 1000.0, 1940.0, 1200.0], [1900.0, 1000.0, 1940.0, 1200.0]),
     ([1000.0, -20.0, 1200.0, 20.0], [1000.0, -20.0, 1200.0, 20.0]),
     ([1000.0, 1260.0, 1200.0, 1300.0], [1000.0, 1260.0, 1200.0, 1300.0]),
     ([-20.0, -20.0, 20.0, 20.0], [-20.0, -20.0, 20.0, 20.0],),
     ([1900.0, -20.0, 1940.0, 20.0], [1900.0, -20.0, 1940.0, 20.0],),
     ([-20.0, 1260.0, 20.0, 1300.0], [-20.0, 1260.0, 20.0, 1300.0],),
     ([1900.0, 1260.0, 1940.0, 1300.0], [1900.0, 1260.0, 1940.0, 1300.0]),
     ([-40.0, 0.0, -20.0, 10.0], [-40.0, 0.0, -20.0, 10.0]),
     ([1940.0, 0.0, 1960.0, 20.0], [1940.0, 0.0, 1960.0, 20.0],),
     ([0.0, -40.0, 1920.0, -20.0], [0.0, -40.0, 1920.0, -20.0]),
     ([0.0, 1300.0, 1920.0, 1320.0], [0.0, 1300.0, 1920.0, 1320.0],),
     ([-100.0, -100.0, -20.0, -20.0], [-100.0, -100.0, -20.0, -20.0]),
     ([1940.0, 1300.0, 1960.0, 1320.0], [1940.0, 1300.0, 1960.0, 1320.0]),
    ]
)
def test_clip_coords_default_args(test_input, expected):
    """
    Test truncate coordinates results with default arguments.
    """
    # arrange
    correlator = BoxCorrelator()
    x_min, y_min, x_max, y_max = test_input
    # act
    ans = correlator._clip_coords(x_min=x_min,
                                  y_min=y_min,
                                  x_max=x_max,
                                  y_max=y_max)
    # assert
    assert ans == approx(expected)


@pytest.mark.parametrize(
    "test_input, clipping, expected",
    [([0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 0.0, 100.0, 100.0],),  # not truncated
     ([100.0, 50.0, 200.0, 100.0], [0.0, 0.0, 1920.0, 1280.0], [100.0, 50.0, 200.0, 100.0],),  # not truncated
     ([-50.0, 1000.0, 50.0, 1200.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 1000.0, 50.0, 1200.0],),  # sticks out to left
     ([1900.0, 1000.0, 1940.0, 1200.0], [0.0, 0.0, 1920.0, 1280.0], [1900.0, 1000.0, 1920.0, 1200.0],),  # sticks out to right
     ([1000.0, -20.0, 1200.0, 20.0], [0.0, 0.0, 1920.0, 1280.0], [1000.0, 0.0, 1200.0, 20.0]),  # sticks out to top
     ([1000.0, 1260.0, 1200.0, 1300.0], [0.0, 0.0, 1920.0, 1280.0], [1000.0, 1260.0, 1200.0, 1280.0],),  # sticks out to bottom
     ([-20.0, -20.0, 20.0, 20.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 0.0, 20.0, 20.0],),  # sticks out at top left corner
     ([1900.0, -20.0, 1940.0, 20.0], [0.0, 0.0, 1920.0, 1280.0], [1900.0, 0.0, 1920.0, 20.0],),  # sticks out at top right corner
     ([-20.0, 1260.0, 20.0, 1300.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 1260.0, 20.0, 1280.0],),  # sticks out at bottom left corner
     ([1900.0, 1260.0, 1940.0, 1300.0], [0.0, 0.0, 1920.0, 1280.0], [1900.0, 1260.0, 1920.0, 1280.0],),  # sticks out at bottom right corner
     ([-40.0, 0.0, -20.0, 10.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 0.0, 0.0, 10.0],),  # shifted to the left
     ([1940.0, 0.0, 1960.0, 20.0], [0.0, 0.0, 1920.0, 1280.0], [1920.0, 0.0, 1920.0, 20.0]),  # shifted to the right
     ([0.0, -40.0, 1920.0, -20.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 0.0, 1920.0, 0.0],),  # shifted to the top
     ([0.0, 1300.0, 1920.0, 1320.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 1280.0, 1920.0, 1280.0],),  # shifted to the bottom
     ([-100.0, -100.0, -20.0, -20.0], [0.0, 0.0, 1920.0, 1280.0], [0.0, 0.0, 0.0, 0.0],),  # shifted to the bottom left
     ([1940.0, 1300.0, 1960.0, 1320.0], [0.0, 0.0, 1920.0, 1280.0], [1920.0, 1280.0, 1920.0, 1280.0],),  # shifted to the bottom right
    ]
)
def test_clip_coords(test_input, clipping, expected):
    """
    Test truncate coordinates results.
    """
    # arrange
    correlator = BoxCorrelator()
    x_min, y_min, x_max, y_max = test_input
    clip_x_min, clip_y_min, clip_x_max, clip_y_max = clipping
    # act
    ans = correlator._clip_coords(x_min=x_min,
                                  y_min=y_min,
                                  x_max=x_max,
                                  y_max=y_max,
                                  clip_x_min=clip_x_min,
                                  clip_y_min=clip_y_min,
                                  clip_x_max=clip_x_max,
                                  clip_y_max=clip_y_max)
    # assert
    assert ans == approx(expected)
