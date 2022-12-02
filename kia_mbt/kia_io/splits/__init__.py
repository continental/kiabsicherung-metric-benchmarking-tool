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
Imports all constants defining the KIA dataset splits

There are four constants which each contain a List[str] of the sequence names
which are allowed for use in the respective split.

The official splits per tranche per company is defined by:

- `TRAIN_{company}_TRANCHE_{num}`: Can be used to train neural networks.
- `VAL_{company}_TRANCHE_{num}`: Can be used to validate the neural network
  during training.
- `TEST_{company}_TRANCHE_{num}`: This split must not be used during development
  and can only be used (ideally) one time once the development is done.

Note that not for all tranches there are dedicated splits.

In addition there is also one constant of the name {company}_TRANCHE_{num} that
contains all sequence names of the respective tranche.
"""

from kia_mbt.kia_io.splits.custom_splitting import *
from kia_mbt.kia_io.splits.tranche_1_bit import *
from kia_mbt.kia_io.splits.tranche_2_bit import *
from kia_mbt.kia_io.splits.tranche_3_bit import *
from kia_mbt.kia_io.splits.tranche_4_bit import *
from kia_mbt.kia_io.splits.tranche_5_bit import *
from kia_mbt.kia_io.splits.tranche_1_mv import *
from kia_mbt.kia_io.splits.tranche_2_mv import *
from kia_mbt.kia_io.splits.tranche_4_mv import *
from kia_mbt.kia_io.splits.tranche_5_mv import *
from kia_mbt.kia_io.splits.tranche_6_mv import *
from kia_mbt.kia_io.splits.tranche_7_mv import *
