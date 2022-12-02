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
Defines sequences of MV tranche 7 with official dataset splitting.

"""

TRAIN_MV_TRANCHE_7 = [
    "mv_results_sequence_0083_1d2b8ce833854587928e2a0d2e38ae46",
    "mv_results_sequence_0090_d451639322d144a7b7d3b8bcfc4b681d",
    "mv_results_sequence_0091_5b55471851cb441091578854dfa9da56",
]

VAL_MV_TRANCHE_7 = ["mv_results_sequence_0095_d26cfb610d064747b4599a1f2e150aa2"]

TEST_MV_TRANCHE_7 = [
    "mv_results_sequence_0084_33190a04594547f3b126ec5d7be1ac8d",
    "mv_results_sequence_0085_c3c573057ae34c47b003d5a4ca8fbc71",
    "mv_results_sequence_0086_784de372be1a4629bb9f7bc9251c1645",
    "mv_results_sequence_0087_a90b28605c3b4c8ca1f62a271e082c5d",
    "mv_results_sequence_0088_60dae98803fc4ad7bc9f51e023c6a1e6",
    "mv_results_sequence_0089_6c1eeba5f5b84791a56e560bf27e86b2",
    "mv_results_sequence_0092_373bc859a41f4ae99b6ef3cdde9f3975",
    "mv_results_sequence_0093_f377cafae31a450d883d6b0ea860dbdb",
    "mv_results_sequence_0094_804ccde1d7a447df8012915ba873154b",
    "mv_results_sequence_0096_86a1c4741e7c49ef9286db7f5a4413bb",
]

MV_TRANCHE_7 = TRAIN_MV_TRANCHE_7 + VAL_MV_TRANCHE_7 + TEST_MV_TRANCHE_7
