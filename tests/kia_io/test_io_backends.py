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
This file contains the test for the KIA Dataset backends.

"""

import unittest
from PIL import ImageStat
from kia_mbt.kia_io.fs_backend import KIADatasetFSBackend


class TestMinioBackend(unittest.TestCase):
    """
    This class implements a unit test for the backends for the KIA dataset
    loader.
    """

    def setUp(self) -> None:
        """
        Setup
        """

        # create file system backed
        self.fs_backend = KIADatasetFSBackend("/mnt/share/kia/data")

        # some test tokens
        self.test_token_true = "bit_results_sequence_0211-fb32183497c34de4b9696aa3c3a48640/sensor/camera/left/png/arb-camera136-0211-fb32183497c34de4b9696aa3c3a48640-0135.png"
        self.test_token_false = "bit_results_sequence_0211-fb32183497c34de4b9696aa3c3a48640/sensor/camera/right/png/arb-camera136-0211-fb32183497c34de4b9696aa3c3a48640-0135.png"
        self.test_token_json = "bit_results_sequence_0211-fb32183497c34de4b9696aa3c3a48640/ground-truth/2d-bounding-box-fixed_json/arb-camera136-0211-fb32183497c34de4b9696aa3c3a48640-0135.json"

    def test_get_object_names(self) -> None:
        """
        Test method for getting all object names

        Parameters
        ----------
            client : KIADatasetMinIOBackend
                MinIO backend object
        """

        objects_filenames = self.fs_backend.get_object_names()
        self.assertGreater(len(objects_filenames), 0)

    def test_exists_object_name(self) -> None:
        """
        Test method for the exists object name method

        Parameters
        ----------
            client : KIADatasetMinIOBackend
                MinIO backend object
        """

        exists_true = self.fs_backend.exists_object_name(self.test_token_true)
        exists_false = self.fs_backend.exists_object_name(self.test_token_false)
        self.assertTrue(exists_true)
        self.assertFalse(exists_false)

    def test_get_image_object(self) -> None:
        """
        Test method for getting an image object

        Parameters
        ----------
            client : KIADatasetMinIOBackend
                MinIO backend object
        """

        img = self.fs_backend.get_image_object(self.test_token_true)
        img_stats = ImageStat.Stat(img)
        self.assertEqual(img.width, 1920)
        self.assertEqual(img.height, 1280)
        self.assertAlmostEqual(img_stats.mean[0], 125.90232788)
        self.assertAlmostEqual(img_stats.mean[1], 133.86449015)
        self.assertAlmostEqual(img_stats.mean[2], 155.43444213)

    def test_get_json_object(self) -> None:
        """
        Test method for getting a JSOn object

        Parameters
        ----------
            client : KIADatasetMinIOBackend
                MinIO backend object
        """

        detections_2d = self.fs_backend.get_json_object(self.test_token_json)
        self.assertEqual(detections_2d["1722"]["c_x"], 1537)


if __name__ == "__main__":
    unittest.main()
