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
Test coordinate conversion.

"""

from pytest import approx

from kia_mbt.kia_correlate.box_correlator import BoxCorrelator


def test_convert_coords_int():
    """
    Test coordinate conversion with integer input.
    """
    # arrange
    correlator = BoxCorrelator()
    # initialize box
    center = (960, 540)  # c_x, c_y
    size = (100, 50)  # widht, height
    # act
    coords = correlator._convert_coords(center=center,
                                        size=size)
    x_min, y_min, x_max, y_max = coords
    # assert
    assert x_min == approx(910, abs=1e-12)
    assert y_min == approx(515, abs=1e-12)
    assert x_max == approx(1010, abs=1e-12)
    assert y_max == approx(565, abs=1e-12)
