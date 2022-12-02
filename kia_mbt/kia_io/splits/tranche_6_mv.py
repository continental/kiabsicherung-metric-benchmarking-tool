# Copyright (c) 2022 Continental AG and subsidiaries.
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
Defines sequences of MV tranche 6 with official dataset splitting.

"""

TRAIN_MV_TRANCHE_6 = [
    "mv_results_sequence_0066_93ae77d052394a5eb3b03aab5c9c3c14",
    "mv_results_sequence_0067_e78d51ef67fa448db0ca6387b7366f50",
]

VAL_MV_TRANCHE_6 = []

TEST_MV_TRANCHE_6 = [
    "mv_results_sequence_0065_a22915a1081d44518e1916b85417fc07",
    "mv_results_sequence_0068_db25e2b8ee2d4058aac0277211b077e1",
    "mv_results_sequence_0069_a802056d2d0c49399f8adb7c81ee2b04",
    "mv_results_sequence_0070_6c33cea450f745e38c156a8d13d2fad3",
    "mv_results_sequence_0071_f0a292d2f8da45adb9d462bd9a3c0e60",
    "mv_results_sequence_0072_050c2b2d45af4ffabfdb1b28bea9e26c",
    "mv_results_sequence_0073_587e46d660c642e0bf61f8e6376b1ba9",
    "mv_results_sequence_0074_9b7acfc0bf5d477280320a66fa3f6f49",
    "mv_results_sequence_0075_1207ac3ea0b3473484f0e417dc6b5e66",
    "mv_results_sequence_0076_588681de7605446fb6f68f570227cbfa",
]

TEST_MV_TRANCHE_6_DOMAIN_ADAPTATION = [
    "mv_results_sequence_0077_f42cd52e06244d63b9a9a6a7aa9e2fd1",
    "mv_results_sequence_0078_70f0f5bb6a9f4f7b970a149906f686b3",
    "mv_results_sequence_0079_f21202d6d8cd48839189d2457acbd489",
    "mv_results_sequence_0080_683db0458dd048f48430dfdfcc3d709b",
    "mv_results_sequence_0081_a91c9c209cd146ba862e229097a19772",
    "mv_results_sequence_0082_4e94fbd288a640dfb29b7805c634e7e1",
]

MV_TRANCHE_6 = (
    TRAIN_MV_TRANCHE_6
    + VAL_MV_TRANCHE_6
    + TEST_MV_TRANCHE_6
    + TEST_MV_TRANCHE_6_DOMAIN_ADAPTATION
)
