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
Defines sequences of BIT-TS tranche 5 with official dataset splitting.

"""

TRAIN_BIT_TRANCHE_5_DYNAMIC = [
    "bit_results_sequence_0263-c372600fd89a45d188a9c664b5ebbed7",
]

VAL_BIT_TRANCHE_5_DYNAMIC = [
    "bit_results_sequence_0250-018426edb1af4f6aaf85bd08e86e4fbc",
]

TEST_BIT_TRANCHE_5_DYNAMIC = [
    "bit_results_sequence_0251-253e6b8cbeed4395bab7f2948eb9fd81",
    "bit_results_sequence_0252-0f5feb086bb444bfaf872ede1f733cef",
    "bit_results_sequence_0301-1d3cff0469f546e0ab996031cec8375b",
    "bit_results_sequence_0302-7e6ab39962ec4b02a5a764b9226132a7",
    "bit_results_sequence_0303-8cfd2b8a8a6146a399403bcad690cc46",
    "bit_results_sequence_0484-2042f3ebfc7948d5bd6b1b8823aef556",
]

TRAIN_BIT_TRANCHE_5_STATIC = [
    "bit_results_sequence_0270-e58a46e8ba634a66899cf1bb1d2f1e0b",
]

VAL_BIT_TRANCHE_5_STATIC = []

TEST_BIT_TRANCHE_5_STATIC = [
    "bit_results_sequence_0264-adfde56a1b1f449798b556eb55925caa",
    "bit_results_sequence_0265-045ade553db44e31ad5fc2f625f866a5",
    "bit_results_sequence_0271-a52901e61f1b499b9d42469ab9463393",
    "bit_results_sequence_0272-aa1940558efa4b5ea6987a776c5bae84",
    "bit_results_sequence_0273-5f4b1ef966ce45228b92239f6de8a9ba",
    "bit_results_sequence_0310-348fd27a157e45f4b61c27221e6a585a",
    "bit_results_sequence_0311-461fdc066cc24623b9aad0ce22d25e90",
    "bit_results_sequence_0312-d7f794f950504b329f12c3b7480a2c6d",
    "bit_results_sequence_0320-dd7e7f6159f543fd97d5333bd2a44261",
    "bit_results_sequence_0321-62888640c8184505ac3c9a133bad984e",
    "bit_results_sequence_0322-b6bbbebf253e4808abf57f0c30fb5b34",
]

TRAIN_BIT_TRANCHE_5 = TRAIN_BIT_TRANCHE_5_DYNAMIC + TRAIN_BIT_TRANCHE_5_STATIC
VAL_BIT_TRANCHE_5 = VAL_BIT_TRANCHE_5_DYNAMIC + VAL_BIT_TRANCHE_5_STATIC
TEST_BIT_TRANCHE_5 = TEST_BIT_TRANCHE_5_DYNAMIC + TEST_BIT_TRANCHE_5_STATIC

BIT_TRANCHE_5 = TRAIN_BIT_TRANCHE_5 + VAL_BIT_TRANCHE_5 + TEST_BIT_TRANCHE_5
