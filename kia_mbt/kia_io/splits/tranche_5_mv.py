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
Defines sequences of MV tranche 5 with official dataset splitting.

"""

TRAIN_MV_TRANCHE_5 = [
    "mv_results_sequence_0053_849d9b8b2c78442c8d3d81562a1ad10a",
    "mv_results_sequence_0056_3b99df6d380448e5ae94386a502ed1ed",
    "mv_results_sequence_0061_f32d780a2fc84b6db3e5d94337c2ba76",
]

VAL_MV_TRANCHE_5 = [
    "mv_results_sequence_0057_56648521b41744be93a4d2b94a6d9432",
]

TEST_MV_TRANCHE_5 = [
    "mv_results_sequence_0052_b9f1277e3fb6499695bca98d88ce8e4e",
    "mv_results_sequence_0054_357d946f054b48ce9fc43c1d47183be2",
    "mv_results_sequence_0055_1dab4c8b18934b2499fdd1df10d4a91c",
    "mv_results_sequence_0058_eba9f412b55746a292ceaf90cbea8d36",
    "mv_results_sequence_0059_010b349bb4c643a9bca420413605d878",
    "mv_results_sequence_0060_1597c400387847048e102505d2e7f8ad",
    "mv_results_sequence_0062_acbcdc0eb23743869b8a78b1e7ac168b",
    "mv_results_sequence_0063_ed44fc840df6421e9a7e41bd30b2950c",
    "mv_results_sequence_0064_224b973925d84f208a377fda185d842f",
]

MV_TRANCHE_5 = TRAIN_MV_TRANCHE_5 + VAL_MV_TRANCHE_5 + TEST_MV_TRANCHE_5
