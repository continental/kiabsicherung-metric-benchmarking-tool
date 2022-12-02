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
This file contains the output formatter for KI Absicherung.

"""

from typing import List, Dict, Tuple
import datetime
import json
import pandas as pd


class KiaFormatter:
    """
    Formatter for the KI Absicherung data.

    Attributes
    ----------
        _version : str
            Version used for file header.
        _tool : str
            Tool name used for file header.
        _time : datetime.datetime
            Timestamp used for file header.

    """

    def __init__(self, version: str, tool: str):
        """
        Setup of formatter.

        Parameters
        ----------
            version : str
                Tool version.
            tool : str
                Tool name.

        """
        self._version = version
        self._tool = tool
        self._time = datetime.datetime.now()

        # generate version entry
        self._version_entry = self._format_version_entry(
            version=self._version, tool=self._tool, time=self._time
        )

    def format_global_metrics(
        self, metric_results: List[Tuple[int, str, pd.DataFrame]]
    ) -> str:
        """
        Format global metrics to JSON string.

        Parameters
        ----------
            metric_results : List[Tuple[int, str, pd.DataFrame]]

        Returns
        -------
            JSON string for global metrics.

        """
        # format single metrics
        formatted_metrics = list()

        # iterate over metrics
        for metric_entry in metric_results:
            formatted_metrics.append(self._format_global_metric(metric_entry))

        # combine metrics into one dictionary
        output_str = self._combine_global_metrics(formatted_metrics=formatted_metrics)

        # dump dictionary as JSON string
        return output_str

    def format_per_sample_metrics(
        self, metric_results: List[Tuple[int, str, pd.DataFrame]]
    ) -> Dict[str, str]:
        """
        Format per sample metrics to dictionary containing JSON strings.

        The dictionary has the sample names as keys and the JSON string
        representations of the metrics in TP1/TP3 annotation format as
        values.

        Parameters
        ----------
            metric_results : List[Tuple[int, str, pd.DataFrame]]

        Returns
        -------
            JSON strings for per sample metrics.

        """
        # format single metrics
        formatted_metrics = list()

        # iterate over metrics
        for metric_entry in metric_results:
            formatted_metrics.append(self._format_per_sample_metric(metric_entry))

        # combine metrics into dictionary containing n_sample many combined JSON strings.
        output_strings = self._combine_per_sample_metrics(
            formatted_metrics=formatted_metrics
        )

        # dump dictionary containing n_samples many JSON strings
        return output_strings

    def _format_version_entry(
        self, version: str, tool: str, time: datetime.datetime
    ) -> Dict:
        """
        Create the version header.

        The version header consists of a version, a tool and the timestep when
        the tool has added some new data. The timestamp is created when this
        method is called by the current time.

        Parameters
        ----------
            version : str
                The version of the tool.
            tool : str
                The name of the tool.
            time : datetime.datetime
                The timestamp to use.

        Returns
        -------
            Version entry dictionary.

        """
        version_entry = {
            "__version_entry__": [
                {
                    "__Version__": version,
                    "__Tool__": tool,
                    "__Time__": time.strftime("%a, %d %b %Y %H:%M:%S %z"),
                }
            ]
        }
        return version_entry

    def _format_metric_id(self, identifier: str) -> str:
        """
        Helper function to format a metric identifier.

        Parameters
        ----------
            identifier : int
                The identifier of the metric as integer.

        Returns
        -------
            Metric identifier as formatted string representation.

        """
        return "__mtrc" + str(identifier).zfill(5) + "__"

    def _format_metric_name(self, name: str) -> str:
        """
        Helper function to format a metric name.

        Parameters
        ----------
            name : str
                Human-readable name of the metric.

        Returns
        -------
            Metric name as formatted string.

        """
        return name.replace(" ", "_")

    def _format_global_metric(
        self, metric_entry: Tuple[int, str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Format a global metric.

        The output dictionary has __mtrc<id>__ as key and
        {name: <metric_name, value: <metric_value>} as value.

        Parameters
        ----------
            metric_frame : pd.DataFrame

        Returns
        -------
            A dictionary containing the formatted metric.

        """
        # unpack metric_entry (currently tuple with three entries)
        metric_id = metric_entry[0]
        metric_name = metric_entry[1]
        metric_data = metric_entry[2]

        # gets pandas frame for single metric with just one row, possibly multiple columns
        value = metric_data.to_dict(orient="records")[0]

        output_dict = {
            self._format_metric_id(identifier=metric_id): {
                "name": self._format_metric_name(name=metric_name),
                "value": value,
            }
        }
        return output_dict

    def _format_per_sample_metric(
        self, metric_result: Tuple[int, str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Format a per-sample metric to a dictionary.

        The output dictionary has <sample_name> as key and
        {__mtrc<id>__: {name: <metric_name, value: <metric_value>}} as value.

        Parameters
        ----------
            metric_result : Tuple[int, str, pd.DataFrame]

        Returns
        -------
            A dictionary containing the formatted metric dicts.

        """
        # unpack metric_result (currently tuple with three entries)
        metric_id = metric_result[0]
        metric_name = metric_result[1]
        metric_data = metric_result[2]

        # gets pandas frame with #n_samples many rows, possibly multiple columns
        sample_values = metric_data.to_dict(orient="index")

        outputs = dict()
        for sample_name, sample_met in sample_values.items():
            output_dict = {
                self._format_metric_id(identifier=metric_id): {
                    "name": self._format_metric_name(name=metric_name),
                    "value": sample_met,
                }
            }
            outputs[sample_name] = output_dict

        return outputs

    def _combine_global_metrics(self, formatted_metrics: List[Dict]) -> str:
        """
        Combine formatted metrics into a string representation.

        This method combines multiple formatted metrics into one string
        representation.

        Parameters
        ----------
            formatted_metrics : List
                A list of dictionaries containing the formatted metrics as dicts.

        Returns
        -------
            Combined formatted string representation of all formatted dictionaries.

        """
        combined_dict = dict()

        combined_dict.update(self._version_entry)

        for metric in formatted_metrics:
            # insert the metric with key '__mtrc<id>__'
            for key, met in metric.items():
                combined_dict[key] = met

        output_str = json.dumps(combined_dict, indent=4)
        return output_str

    def _combine_per_sample_metrics(
        self, formatted_metrics: List[Dict]
    ) -> Dict[str, str]:
        """
        Combine formatted metrics into a string representation

        This method combines multiple formatted metrics into dictionary
        with <sample_name> as keys and JSON string representations as values.

        Parameters
        ----------
            formatted_metrics: List[Dict]

        Returns
        -------
            Dictionary with per sample combined string representations of all formatted dictionaries.

        """
        output_strings = dict()
        sample_names = list(
            formatted_metrics[0].keys()
        )  # TODO: not safe for empty list

        # iterate over metrics
        for sample_name in sample_names:
            # iterate over metrics
            combined_sample_dict = dict()
            combined_sample_dict.update(self._version_entry)

            for metric in formatted_metrics:
                # build one dictionary of all metrics for one sample
                sample_metric = metric.get(sample_name, None)
                # insert the metric with key '__mtrc<id>'
                for key, met in sample_metric.items():
                    combined_sample_dict[key] = met
                # dump json string for all metrics of one sample
                sample_str = json.dumps(combined_sample_dict, indent=4)
                output_strings[sample_name] = sample_str

        return output_strings
