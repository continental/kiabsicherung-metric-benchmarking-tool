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
This file contains the interface class for the KIA dataset backend.

"""

from typing import List
from abc import ABC, abstractmethod
from PIL import Image


class KIADatasetBackend(ABC):
    """
    Interface class for the KIA Dataset backends.

    A new backend can be implemented by deriving from this class and
    implementing all defined abstract methods.
    """

    @abstractmethod
    def get_object_names(self, sequence: str = "") -> List[str]:
        """
        Get all object names

        This method returns all object names of the KIA dataset. An object name
        is the relative path from origin to an object or a file in the dataset
        in the folder structure of the original distribution.
        The path separators must be ``/``, and the paths should be normalized
        (see ``os.path.normpath``).

        Parameters
        ----------
            sequence : str
                If a sequence name is given, only object names of this sequence
                will be returned.

        Returns
        -------
        A list of strings containing the object names.
        """

        pass

    @abstractmethod
    def exists_object_name(self, object_name: str) -> bool:
        """
        Test if an object name exists

        For the given object name it shall be tested if it exists or not in the
        dataset.

        Parameters
        ----------
            object_name : str
                Object or file name

        Returns
        -------
        Returns True of the object name exists, otherwise False.
        """

        pass

    @abstractmethod
    def get_image_object(self, object_name: str) -> Image.Image:
        """
        Get an image object

        This method loads an image object, e.g. a PNG file, and returns it as a
        PIL Image.

        Parameters
        ----------
            object_name : str
                Object or file name

        Returns
        -------
        Loaded image object.
        """

        pass

    @abstractmethod
    def get_json_object(self, object_name: str):
        """
        Get an JSON object

        This method loads an JSON object and returns it as an dictionary.

        Parameters
        ----------
            object_name : str
                Object or file name

        Returns
        -------
        Loaded image object.
        """

        pass
