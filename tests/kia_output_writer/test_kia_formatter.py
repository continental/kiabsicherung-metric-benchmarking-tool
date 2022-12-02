# Copyright (c) 2022 Elektronische Fahrwerksysteme GmbH (www.efs-auto.com).
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
Unit test for kia_formatter.

"""

from tests.kia_output_writer.conftest import get_empty_global_metric_data, get_per_sample_metric_test_data
from tests.kia_output_writer.conftest import get_global_metric_test_data
from kia_mbt.kia_output_writer.kia_formatter import KiaFormatter


def test_kia_formatter_init():
    """
    Test KiaFormatter initialization.
    """
    # arrange
    version = 'v01'
    tool = 'MetricBenchmarkTool'
    kia_formatter = KiaFormatter(version=version,
                                 tool=tool)
    # assert
    assert kia_formatter._version == version
    assert kia_formatter._tool == tool
    assert kia_formatter._time
    assert kia_formatter._version_entry


def test_format_global_metric():
    """
    Test formatting of single global metric in _format_global_metric.
    """
    # arrange
    global_metrics = get_global_metric_test_data()
    kia_formatter = KiaFormatter(version='v01', tool='mbt')
    # act
    ans0 = kia_formatter._format_global_metric(metric_entry=global_metrics[0])
    ans1 = kia_formatter._format_global_metric(metric_entry=global_metrics[1])
    # assert
    assert '__mtrc01030__' in ans0
    assert len(ans0) == 1
    assert '__mtrc01031__' in ans1
    assert len(ans1) == 1


def test_combine_global_metrics():
    """
    Test combining global metrics in _combine_global_metrics.
    """
    # arrange
    kia_formatter = KiaFormatter(version='v01', tool='mbt')

    global_metrics = get_global_metric_test_data()
    formatted_metrics = list()
    for metric_entry in global_metrics:
        formatted_metrics.append(kia_formatter._format_global_metric(metric_entry))
    # act
    output_str = kia_formatter._combine_global_metrics(formatted_metrics)
    # assert
    assert isinstance(output_str, str)


def test_combine_per_sample_metrics():
    """
    Test combining per sample metrics in _combine_per_sample_metrics.
    """
    # arrange
    kia_formatter = KiaFormatter(version='v01', tool='mbt')

    sample_metrics = get_per_sample_metric_test_data()
    formatted_metrics = list()
    for metric_entry in sample_metrics:
        formatted_metrics.append(kia_formatter._format_per_sample_metric(metric_entry))
    # act
    output_strings = kia_formatter._combine_per_sample_metrics(formatted_metrics)
    # assert
    assert isinstance(output_strings, dict)


def test_format_version_entry():
    """
    Test version header formatting.
    """
    pass


def test_format_metric_id():
    """
    Test metric id formatting.
    """
    pass


def test_format_metric_name():
    """
    Test metric name formatting.
    """
    pass
