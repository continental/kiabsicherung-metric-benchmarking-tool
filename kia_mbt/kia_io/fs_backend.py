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
This file contains the a file system backend.

"""

import os
import glob
from typing import List
from kia_mbt.kia_io.backend import KIADatasetBackend
from PIL import Image
import json


class KIADatasetFSBackend(KIADatasetBackend):
    """
    This class implements a backend for file storages.
    """

    def __init__(self, data_folder: str) -> None:
        """
        Creates the file system backend and checks if the data path is
        available.

        Parameters
        ----------
            data_folder : str
                The path to the root folder, where the data is stored.
        """

        # Check if data folder exists

        # Store data folder
        self.data_folder = data_folder.replace(os.sep, "/")

    def get_object_names(self, sequence: str = "") -> List[str]:
        """
        Get all object names

        This method returns all object names as relative pathes to the folder
        containing the KIA Dataset.

        Parameters
        ----------
            sequence : str
                If a sequence name is given, only object names of this sequence
                will be returned.

        Returns
        -------
        A list of relative pathes to all objects as strings.
        """

        prefix = self.data_folder
        if sequence:
            prefix = os.path.join(self.data_folder, sequence, "").replace(os.sep, "/")

        objects = []
        for filename in glob.iglob(os.path.join(prefix, "**/*.*"), recursive=True):
            filename = filename.replace(os.sep, "/")
            filename = filename[len(self.data_folder) :]
            if filename.startswith("/"):
                filename = filename[len("/") :]
            objects.append(filename)

        return objects

    def exists_object_name(self, object_name: str) -> bool:
        """
        Test if an object name exists.

        Parameters
        ----------
            object_name : str
                Object name as relative path to the file

        Returns
        -------
        True if it exists, otherwise False.
        """

        return os.path.exists(
            os.path.join(self.data_folder, object_name).replace(os.sep, "/")
        )

    def get_image_object(self, object_name: str) -> Image.Image:
        """
        Get image object from file.

        Parameters
        ----------
            object_name : str
                Object name as relative path to the file

        Returns
        -------
        Loaded image object as PIL Image.
        """

        return Image.open(
            os.path.join(self.data_folder, object_name).replace(os.sep, "/")
        )

    def get_json_object(self, object_name: str):
        """
        Get JSON object from file.

        Parameters
        ----------
            object_name : str
                Object name as relative path to the file

        Returns
        -------
        Loaded JSON object as dictionary.
        """

        with open(
            os.path.join(self.data_folder, object_name).replace(os.sep, "/"), "r"
        ) as file:
            return json.load(file)
