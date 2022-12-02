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
Defines sequences of BIT-TS tranche 4 with official dataset splitting.

"""

TRAIN_BIT_TRANCHE_4_DYNAMIC = [
    "bit_results_sequence_0149-4c0e36fef9394df0bc7558a7187fb53f",
    "bit_results_sequence_0150-3bbbcf77421040139102b786c1026f24",
    "bit_results_sequence_0151-fc112ac1b90f4b9e9221543460d5eac0",
    "bit_results_sequence_0152-18a097b800704cd2ba4ae4a937820f13",
    "bit_results_sequence_0153-97588181ad4f42f49c11cd2264e5fa01",
    "bit_results_sequence_0154-aae134a550c2462fa2dd5b87619c73db",
    "bit_results_sequence_0155-c5885dc8f0174b038ecd7416bb68a4e4",
    "bit_results_sequence_0156-e4cfa019fdda41a19fdf60b5f6e7d981",
]
VAL_BIT_TRANCHE_4_DYNAMIC = [
    "bit_results_sequence_0147-4d53650ffc4a49909671fd74ed6beec0",
    "bit_results_sequence_0148-d71bbded97534c09bd7e10af03c37323",
]
TEST_BIT_TRANCHE_4_DYNAMIC = []

TRAIN_BIT_TRANCHE_4_STATIC = [
    "bit_results_sequence_0177-f256ff87158c40a4bded781bcd427d60",
    "bit_results_sequence_0178-2f997f13c6474d45b338ad67644a1b2d",
    "bit_results_sequence_0179-21032ac691f24ce087ab3c4cc3a0b5fc",
    "bit_results_sequence_0180-060fba0d64744724aafe39a1e45fc6e1",
    "bit_results_sequence_0181-b79af3034b54420ab93c5bbe8d8009ca",
    "bit_results_sequence_0182-1c488d65f3af4b73bb81e12c26209730",
    "bit_results_sequence_0184-4ee2861981f44690b12c42aedd7b7359",
    "bit_results_sequence_0185-195e048967e24bcdb709dc3d803e9e55",
    "bit_results_sequence_0187-c29cb10799304d7ca459483d51701e54",
    "bit_results_sequence_0192-4fcdd040d5264cf08fa9ad4fce6313f7",
    "bit_results_sequence_0193-c01415c9c5a940a4acdfd09c6bcb2de5",
    "bit_results_sequence_0194-bb51bd95675c466baba0a399d421d55f",
    "bit_results_sequence_0211-fb32183497c34de4b9696aa3c3a48640",
    "bit_results_sequence_0212-68787ea2d23c48daa21c8792987fd8eb",
    "bit_results_sequence_0213-3a35e69cdd98464a9aa6ed841c56a5c8",
    "bit_results_sequence_0214-882545f9069042d4b44e9551bdcbe2ef",
    "bit_results_sequence_0215-d40c77b1bda64aedb87a32a79e450833",
    "bit_results_sequence_0216-f309d24fe71c460a82f6be1d47af3606",
]
VAL_BIT_TRANCHE_4_STATIC = [
    "bit_results_sequence_0171-ba9162a2b5af48c6a94e9ab99ef658bd",
    "bit_results_sequence_0172-d13f8b0cf1b84c6498512bb99e369a36",
    "bit_results_sequence_0173-073d14d72577491abb76e865ce119c91",
]
TEST_BIT_TRANCHE_4_STATIC = [
    "bit_results_sequence_0174-54c7c84860b442eca995b153754b8c37",
    "bit_results_sequence_0209-322405fa2a264f4499ffbf93e5ee17e0",
]

TRAIN_BIT_TRANCHE_4 = TRAIN_BIT_TRANCHE_4_DYNAMIC + TRAIN_BIT_TRANCHE_4_STATIC
VAL_BIT_TRANCHE_4 = VAL_BIT_TRANCHE_4_DYNAMIC + VAL_BIT_TRANCHE_4_STATIC
TEST_BIT_TRANCHE_4 = TEST_BIT_TRANCHE_4_DYNAMIC + TEST_BIT_TRANCHE_4_STATIC

BIT_TRANCHE_4 = TRAIN_BIT_TRANCHE_4 + VAL_BIT_TRANCHE_4 + TEST_BIT_TRANCHE_4
