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
Defines a function to create a custom dataset split.

"""

from typing import List

# import BIT TS dataset splits
from kia_mbt.kia_io.splits.tranche_1_bit import BIT_TRANCHE_1
from kia_mbt.kia_io.splits.tranche_2_bit import BIT_TRANCHE_2
from kia_mbt.kia_io.splits.tranche_3_bit import BIT_TRANCHE_3
from kia_mbt.kia_io.splits.tranche_4_bit import BIT_TRANCHE_4
from kia_mbt.kia_io.splits.tranche_5_bit import BIT_TRANCHE_5

# import MV dataset splits
from kia_mbt.kia_io.splits.tranche_1_mv import MV_TRANCHE_1
from kia_mbt.kia_io.splits.tranche_2_mv import MV_TRANCHE_2
from kia_mbt.kia_io.splits.tranche_4_mv import MV_TRANCHE_4
from kia_mbt.kia_io.splits.tranche_5_mv import MV_TRANCHE_5
from kia_mbt.kia_io.splits.tranche_6_mv import MV_TRANCHE_6
from kia_mbt.kia_io.splits.tranche_7_mv import MV_TRANCHE_7


def create_split(company: str, sequences: List[int]) -> List[str]:
    """
    Manually create a custom split given the sequences.

    Parameters
    ----------
        company : str
            The company for which the split should be created.
        sequences : List[int]
            A list of ints representing the sequnece ids that should be taken.

    Returns
    -------
    A list of strings representing the sequence names as in the other official
    splits.
    """

    elements = (
        BIT_TRANCHE_1
        + BIT_TRANCHE_2
        + BIT_TRANCHE_3
        + BIT_TRANCHE_4
        + BIT_TRANCHE_5
        + MV_TRANCHE_1
        + MV_TRANCHE_2
        + MV_TRANCHE_4
        + MV_TRANCHE_5
        + MV_TRANCHE_6
        + MV_TRANCHE_7
    )
    results = []
    for elem in elements:
        if elem.startswith(company) and int(elem.split("_")[3][:4]) in sequences:
            results.append(elem)
    return sorted(results)
