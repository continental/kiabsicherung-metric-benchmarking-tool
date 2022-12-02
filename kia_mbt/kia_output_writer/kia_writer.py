# Copyright (c) 2022 Elektronische Fahrwerksysteme GmbH (www.efs-auto.com) and
# Continental AG and subsidiaries.
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
This file contains the output writer for the metric data

"""

from typing import Tuple, List, Union
import os
import json
import pandas as pd

from kia_mbt.kia_output_writer import constants
from kia_mbt.kia_output_writer.kia_formatter import KiaFormatter


class KIAWriter:
    """
    Output writer for KIA data.

    Attributes
    ----------
        _company_name : str
            Company name.
        _tool : str
            Tool identifer.
        _release : str
            Software release.
        _commit_id : str
            Git commit id.
        _version : str
            Version string.
        _formatter : KiaFormatter
            The formatter object that is used to format the data.
        _eval_folder : str
            Evaluation folder.
        _path_prefix : str
            Prefix for all object paths.

    """

    def __init__(
        self,
        version_fpath: Union[str, None] = None,
        backend_path: Union[str, None] = None,
    ):
        """
        Setup of the KIA output writer.

        All results will be written into a common folder, which is called
        path_prefix and is of the following format

            <eval_folder>/<company>-<tool>-<release>-<commit-id>

        Thereby, <eval_folder> is the name of the evaluation folder, <company>
        is the name of the company providing the tool, <tool> is the tool name
        in lower case, <release> is the current release version and <commit-id>
        is the git commit id that is used to produce the results.

        Parameters
        ----------
            version_fpath : str
                Path to file containing version information provided in file headers.
            backend_path : str
                Backend file path.

        """
        # initialize version information
        try:
            if version_fpath is None:
                version_fpath = ""
            with open(version_fpath) as v_fpath:
                version_info = json.load(v_fpath)

            self._company_name = version_info.get("COMPANY_NAME", "company")
            self._tool = version_info.get("TOOL", "tool")
            self._release = version_info.get("RELEASE", "r0")
            self._commit_id = version_info.get("COMMIT_ID", "commit_id")
            self._version = version_info.get("VERSION", "v0.0")

        except IOError:
            print("Version file could not be found, using defaults")
            self._company_name = "company"
            self._tool = "tool"
            self._release = "r0"
            self._commit_id = "commit_id"
            self._version = "v0.0"

        # KIA formatter
        self._formatter = KiaFormatter(version=self._version, tool=self._tool)

        # build folder prefix
        self._backend_path = backend_path
        if self._backend_path is None:
            self._backend_path = ""
        self._eval_folder = constants.FOLDER_EVAL
        self._path_prefix = self._get_path_prefix(
            backend_path=self._backend_path,
            tool=self._tool,
            release=self._release,
            commit_id=self._commit_id,
        )

    def write_global_metrics(
        self, global_metrics: List[Tuple[int, str, pd.DataFrame]]
    ) -> None:
        """
        Write global metric data to storage.

        A global metric contains data that is obtained by using some or
        all samples of a dataset. Uses the formatter to format the metric
        data before writing.

        The filepath is the following:
            <path_prefix>/<annotation_folder>

        Parameters
        ----------
            global_metrics : List[Tuple[int, str, pd.DataFrame]]

        """
        output_str = self._formatter.format_global_metrics(global_metrics)

        object_path = os.path.join(
            self._path_prefix, constants.FOLDER_2DBB, self._global_object_name()
        )
        object_path = object_path.replace(os.sep, "/")
        self.write_json(file_path=object_path, json_string=output_str)

    def write_per_sample_metrics(
        self, sample_metrics: Tuple[int, str, pd.DataFrame]
    ) -> None:
        """
        Write sample-based metric data to storage.

        Use the formatter to format the metric data before writing.

        Parameters
        ----------
            sample_metrics : List[Tuple[int, str, pd.DataFrame]]

        """
        output_strings = self._formatter.format_per_sample_metrics(sample_metrics)

        for sample_name, output_str in output_strings.items():
            object_name = self._sample_object_name(sample_name=sample_name)
            object_path = os.path.join(
                self._path_prefix, constants.FOLDER_2DBB, object_name
            )
            object_path = object_path.replace(os.sep, "/")
            self.write_json(file_path=object_path, json_string=output_str)

    def write_json(self, file_path: str, json_string: str) -> None:
        """
        Write JSON string into file.

        Parameters
        ----------
            file_path : str
                Path of the file to write.
            json_string : str
                JSON string to write into file.

        """
        # make directory if not exist
        out_path = file_path.rsplit("/", 1)[0]
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # write JSON string to file
        with open(file_path, "w") as outfile:
            outfile.write(json_string)

    def _global_object_name(self) -> str:
        """
        Generate file name for global metrics file.

        Returns
        -------
            Global object file name.

        """
        return "global_metrics.json"

    def _sample_object_name(self, sample_name: str) -> str:
        """
        Generate file name for per sample metrics file.

        Parameters
        ----------
            sample_name : str

        Returns
        -------
            Sample object file name.

        """
        return sample_name.split("/")[-1] + ".json"

    def _get_path_prefix(
        self, backend_path: str, tool: str, release: str, commit_id: str
    ) -> str:
        """
        Create the path prefix.

        Parameters
        ----------
            tool : str
                Tool name.
            release : str
                Software release.
            commit_id : str
                Git commit id.

        Returns
        -------
            Path prefix.

        """
        folder_name = self._company_name + "-" + tool + "-" + release + "-" + commit_id
        prefix = os.path.join(backend_path, self._eval_folder, folder_name).replace(
            os.sep, "/"
        )
        return prefix
