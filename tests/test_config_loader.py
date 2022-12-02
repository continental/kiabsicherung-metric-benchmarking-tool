import json
import pytest
from kia_mbt.config_loader import *


@pytest.fixture(scope='session')
def config_file(tmp_path_factory):
    config = {
        "io": {
            "data_path":
            "/mnt/share/kia/data",
            "predictions_path":
            "/mnt/share/kia/predictions",
            "results_folder":
            "opelssd-v3...",
            "backend":
            "fs",
            "sequences":
            ["results_sequence_mv_234.....", "results_sequence_mv_076....."]
        },
        "correlate": {
            "iou_threshold": 0.1,
            "matching_type": "complete",
            "clip_truncated_boxes": True,
            "optional_arguments:": {
                "confidence_col": "confidence",
                "annotation_bb_center_col": "center",
                "annotation_bb_size_col": "size",
                "detection_bb_center_col": "center",
                "detection_bb_size_col": "size"
            }
        },
        "filter": {
            "annotation_filter": {
                "class_id == human": ["class_id", "==", "human"],
                "occlusion_est < 0.8": ["occlusion_est", "<", 0.8],
                "within_brake_dist_30kph == True":
                ["within_brake_dist_30kph", "==", True],
                "semantic_area in ['road', 'crossing', 'sidewalk', 'sidewalk_near_crossing']":
                [
                    "semantic_area", "in",
                    ["road", "crossing", "sidewalk_near_crossing"]
                ]
            },
            "prediction_filter": {},
            "matching_filter": {}
        },
        "metrics": {
            "calculate": [1003, 1031],
            "parameters": {
                "1031": {
                    "calculate_per_class": True,
                    "confidence_threshold": None,
                    "confidence_column_name": "confidence",
                    "iou_threshold": None,
                    "iou_column_name": "match_value"
                }
            }
        },
        "writer": {
            "version_file": "version.json",
            "output_path": "/mnt/share/kia"
        }
    }

    fn = tmp_path_factory.mktemp('data') / 'config.json'
    with open(fn, 'w') as fp:
        json.dump(config, fp)
    return fn


def test_config_loader(config_file):
    config_loader = ConfigLoader(config_file)
    io_config = config_loader.get_io_config()
    assert len(io_config.sequences) == 2
    correlate_config = config_loader.get_correlate_config()
    assert correlate_config.iou_threshold == 0.1
    assert len(correlate_config.optional_arguments) == 5
    filter_config = config_loader.get_filter_config()
    assert len(filter_config.annotation_filter) == 4
    assert len(filter_config.matching_filter) == 0
    metric_config = config_loader.get_metric_config()
    assert len(metric_config.calculate) == 2
    metric_param = metric_config.get_metric_parameters(1031)
    assert len(metric_param) == 5
    writer_config = config_loader.get_writer_config()
    assert writer_config.version_file == 'version.json'
