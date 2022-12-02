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
Unit test for kia_writer.

"""

import os
import json
import tempfile

from tests.kia_output_writer.conftest import get_empty_global_metric_data
from tests.kia_output_writer.conftest import get_global_metric_test_data
from tests.kia_output_writer.conftest import get_empty_per_sample_test_data
from tests.kia_output_writer.conftest import get_per_sample_metric_test_data
from kia_mbt.kia_output_writer.kia_writer import KIAWriter
from kia_mbt.kia_output_writer.kia_formatter import KiaFormatter


def test_kia_writer_init():
    """
    Test KiaWriter initialization.
    """
    # arrange
    writer0 = KIAWriter(version_fpath="tests/kia_output_writer/test_version.json")
    writer1 = KIAWriter()
    # assert 0
    assert writer0._company_name == 'TestComp'
    assert writer0._tool == 'TestTool'
    assert writer0._release == 'r1'
    assert writer0._commit_id == 'c123'
    assert writer0._version == 'v0.3'

    assert isinstance(writer0._formatter, KiaFormatter)
    assert writer0._eval_folder
    assert writer0._path_prefix

    # assert 1
    assert writer1._company_name == 'company'
    assert writer1._tool == 'tool'
    assert writer1._release == 'r0'
    assert writer1._commit_id == 'commit_id'
    assert writer1._version == 'v0.0'

    assert isinstance(writer1._formatter, KiaFormatter)
    assert writer1._eval_folder
    assert writer1._path_prefix


def test_kia_writer_write_global_metrics():
    """
    Test KiaWriter write global metrics.
    """
    # arrange
    global_metrics = get_global_metric_test_data()
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_name = temp_dir.name.replace(os.sep, '/')
    writer = KIAWriter(version_fpath="kia_mbt/version.json",
                       backend_path=temp_dir_name)
    # act
    writer.write_global_metrics(global_metrics=global_metrics)
    # assert
    f_path = os.path.join(writer._path_prefix, '2d-bounding-box_json', writer._global_object_name())
    f_path = f_path.replace(os.sep, '/')
    assert os.path.exists(f_path)
    # cleanup
    temp_dir.cleanup()


def test_kia_writer_write_per_sample_metric():
    """
    Tet KiaWriter write per sample metrics.
    """
    # arrange
    sample_metrics = get_per_sample_metric_test_data()
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_name = temp_dir.name.replace(os.sep, '/')
    writer = KIAWriter(version_fpath="kia_mbt/version.json",
                       backend_path=temp_dir_name)
    # act
    writer.write_per_sample_metrics(sample_metrics=sample_metrics)
    # assert
    f_path = os.path.join(writer._path_prefix, '2d-bounding-box_json').replace(os.sep, '/')
    assert os.path.exists(f_path + '/arb-camera001-006d842f-0000.json')
    assert os.path.exists(f_path + '/arb-camera001-006d842f-0001.json')
    assert os.path.exists(f_path + '/arb-camera001-006d842f-0002.json')
    assert os.path.exists(f_path + '/arb-camera001-006d842f-0003.json')
    assert os.path.exists(f_path + '/arb-camera001-006d842f-0005.json')
    # cleanup
    temp_dir.cleanup()


def test_write_json():
    """
    Test KiaWriter write json.
    """
    # arrange
    writer = KIAWriter(version_fpath="tests/kia_output_writer/test_version.json")
    # act
    temp_dir = tempfile.TemporaryDirectory()
    file_path = str(temp_dir.name + '/my_sub' + '/test.json')
    json_string = json.dumps(['abc', ])
    writer.write_json(file_path=file_path,
                      json_string=json_string)
    # assert
    with open(file_path, 'r') as fip:
        res = json.load(fip)
    assert res[0] == 'abc'
    # cleanup
    temp_dir.cleanup()


def test_global_object_name():
    """
    Test KiaWriter _global_object_name.
    """
    # arrange
    writer = KIAWriter(version_fpath="tests/kia_output_writer/test_version.json")
    # act
    ans = writer._global_object_name()
    # assert
    assert ans == 'global_metrics.json'


def test_get_path_prefix():
    """
    Test KiaWriter _get_path_prefix.
    """
    # arrange
    writer = KIAWriter(version_fpath="tests/kia_output_writer/test_version.json")
    # act
    ans = writer._get_path_prefix(backend_path='',
                                  tool='Mbt',
                                  release='r0',
                                  commit_id='c123')
    # assert
    company_name = writer._company_name
    eval_folder = writer._eval_folder
    folder_name = company_name + '-' + 'Mbt' + '-' + 'r0' + '-' + 'c123'
    prefix = eval_folder + '/' + folder_name
    assert ans == prefix
