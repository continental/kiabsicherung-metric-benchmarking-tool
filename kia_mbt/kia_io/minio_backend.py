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

import os
import io
from typing import List
import json
import urllib3
from PIL import Image
from minio import Minio, S3Error
from kia_mbt.kia_io.backend import KIADatasetBackend


class KIADatasetMinIOBackend(KIADatasetBackend):
    """
    This class implements a backend for MinIO storages.
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        endpoint: str,
        bucket_name: str,
        data_folder: str,
        use_proxy: bool = True,
    ) -> None:
        """
        Creates the MinIO client and tests if the bucket can be accessed.

        Parameters
        ----------
            access_key : str
                Access key is like user ID that uniquely identifies your account.
            secret_key : str
                Secret key is the password to your account.
            endpoint : str
                Endpoint of the MinIO service, e.g. some URL address
            bucket_name : str
                Name of the bucket that contains the data
            data_folder : str
                Name of the folder in the bucket that contains the data. This
                can be seen as a prefix.
            use_proxy : bool
                When True (default), a proxy is configured using the environment
                variable http_proxy for the connection to the MinIO service.
                Otherwise, no proxy is used.
        """

        # Create MinIO client and test if bucket can be accessed
        if use_proxy:
            self.minio_client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                http_client=urllib3.ProxyManager(os.environ["http_proxy"]),
            )
        else:
            self.minio_client = Minio(
                endpoint, access_key=access_key, secret_key=secret_key
            )

        # Check if bucket exists
        self.bucket_name = bucket_name
        if not self.minio_client.bucket_exists(self.bucket_name):
            raise ConnectionError("Bucket {} does not exist".format(self.bucket_name))

        # Store data folder
        if not data_folder.endswith("/"):
            data_folder = data_folder + "/"
        self.data_folder = data_folder

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
            prefix = os.path.join(self.data_folder, sequence) + "/"

        objects = self.minio_client.list_objects(
            self.bucket_name, prefix=prefix, recursive=True
        )
        objects_names = []
        for obj in objects:
            # check if object is a file
            if not obj.is_dir:
                # remove prefix, the folder containing the data sequences, and append to list
                objects_names.append(obj.object_name[len(self.data_folder) :])
        return objects_names

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

        try:
            self.minio_client.stat_object(
                self.bucket_name, self.data_folder + object_name
            )
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
        return True

    def get_image_object(self, object_name: str) -> Image.Image:
        """
        Get image object from bucket

        Parameters
        ----------
            object_name : str
                Object name as relative path to the file

        Returns
        -------
        Loaded image object as PIL Image.
        """

        try:
            response = self.minio_client.get_object(
                self.bucket_name, self.data_folder + object_name
            )
            image_object = Image.open(io.BytesIO(response.data))
        finally:
            response.close()
            response.release_conn()
        return image_object

    def get_json_object(self, object_name: str):
        """
        Get JSON object from bucket

        Parameters
        ----------
            object_name : str
                Object name as relative path to the file

        Returns
        -------
        Loaded JSON object as dictionary.
        """

        try:
            response = self.minio_client.get_object(
                self.bucket_name, self.data_folder + object_name
            )
            json_object = json.loads(response.data)
        finally:
            response.close()
            response.release_conn()
        return json_object
