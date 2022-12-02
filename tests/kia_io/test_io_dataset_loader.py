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
This file contains the test for the KIA Dataset Loader.

"""

import unittest
import random
from kia_mbt.kia_io import *


class TestDatasetLoader(unittest.TestCase):
    """
    This class implements a unit test for the backends for the KIA dataset
    loader.
    """

    def setUp(self) -> None:
        """
        Setup function.
        """

        # create file system backed
        fs_backend = KIADatasetFSBackend("/mnt/share/kia/data")
        # create dataset config
        dataset_config_sequences = KIADatasetConfig(sequences=[64])
        self.dataset_config_sequence_names = KIADatasetConfig(
            sequence_names=[
                "bit_results_sequence_0174-54c7c84860b442eca995b153754b8c37",
                "mv_results_sequence_0064_224b973925d84f208a377fda185d842f",
            ]
        )
        # create loader
        self.loader_1 = KIADatasetLoader(fs_backend, dataset_config_sequences)
        self.loader_2 = KIADatasetLoader(fs_backend, self.dataset_config_sequence_names)
        # test sample token
        self.sample_token = (
            "mv/arb-camera001-0064-224b973925d84f208a377fda185d842f-0400"
        )

    def test_get_meta_info(self) -> None:
        """
        Test method for getting additional meta information.
        """

        data = self.loader_1.get_additional_meta_info(self.sample_token)
        self.assertEqual(data.entities[10].world_semantic_area, "sidewalk")
        self.assertEqual(data.light_sources[0].world_elevation, "day")
        self.assertEqual(data.ego_sensors[0].angle_bev_north2fov_deg, 61.39)

    def test_loading_by_sequence_names(self) -> None:
        sample_tokens = self.loader_2.get_sample_tokens()
        test_tokens = random.choices(sample_tokens, k=5)
        for test_token in test_tokens:
            found = False
            sample_token_hash = test_token.split("-")[-2]
            for sequence_name in self.dataset_config_sequence_names.sequence_names:
                if sample_token_hash in sequence_name:
                    found = True
            self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
