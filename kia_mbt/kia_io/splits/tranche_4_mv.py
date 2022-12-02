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
Defines sequences of MV tranche 4 with official dataset splitting.

"""

TRAIN_MV_TRANCHE_4 = [
    "mv_results_sequence_0040_beac809a71b543798474e44bcc61c31d",
    "mv_results_sequence_0041_9d338b0348ca445b9573255f32ac1c1d",
    "mv_results_sequence_0042_ba3e06b52c814854b726d1cd270a32cd",
    "mv_results_sequence_0043_5ca6bdef77c74d2ebb2ee575831ed1a5",
    "mv_results_sequence_0044_bbfe9b85fd1042ae9f7862d27e13604f",
    "mv_results_sequence_0045_2aa99ff8db43437a96769a46d7441af7",
    "mv_results_sequence_0051_dca114b1114e4245a3badbb5f370b6a8",
]

VAL_MV_TRANCHE_4 = [
    "mv_results_sequence_0046_d564d18e4ec14205ab84707fe9366e5c",
]

TEST_MV_TRANCHE_4 = [
    "mv_results_sequence_0047_03f10c4336dc4d85a527ae7e2bfe15f2",
    "mv_results_sequence_0048_3162767e837d4ee18eb2ff2c32186949",
    "mv_results_sequence_0050_caae51aed41c495793865135c856e3bb",
]

MV_TRANCHE_4 = TRAIN_MV_TRANCHE_4 + VAL_MV_TRANCHE_4 + TEST_MV_TRANCHE_4
