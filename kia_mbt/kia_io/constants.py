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
This file contains the constants required for the KIA dataset loader

"""

import kia_mbt.kia_io.splits as splits

### Pathes and folders
FOLDER_PRED = "predictions"
FOLDER_2DBB = "2d-bounding-box_json"
FOLDER_2DBB_FIXED = "2d-bounding-box-fixed_json"
FOLDER_2DBB_ENRICHED = "2d-bounding-box-enriched_json"
FOLDER_META_INFO = "general-globally-per-frame-analysis-enriched_json"

### Dataset annotations
# Mapping of 2D semantic segmentation RGB values and semantic labels
SEMSEG_SEMANTIC_MAPPING = {
    "unlabeled": (0, 0, 0),
    "animal": (100, 90, 0),
    # human classes
    "construction_worker": (220, 20, 200),
    "person": (220, 20, 60),
    "walk_assistance": (220, 20, 100),
    "child": (220, 20, 0),
    "buggy": (220, 20, 175),
    "police_officer": (220, 20, 225),
    "wheelchair_user": (220, 20, 150),
    "cyclist": (255, 64, 64),
    "rider": (255, 0, 0),
    # vehicle classes
    "car": (0, 0, 142),
    "trailer": (0, 0, 110),
    "construction_vehicle": (0, 0, 80),
    "bus": (0, 60, 100),
    "bicycle": (119, 11, 32),
    "truck": (0, 0, 70),
    "motorcycle": (0, 0, 230),
    "police_car": (0, 0, 155),
    "van": (0, 0, 142),
    # movable object classes
    "dynamic": (111, 74, 0),
    # static object classes
    "parking": (250, 170, 160),
    "pole": (153, 153, 153),
    "traffic_light": (250, 170, 30),
    "traffic_sign": (220, 220, 0),
    "ground": (81, 0, 81),
    "lane_marking_bit": (255, 255, 0),
    "lane_marking_mv": (255, 255, 255),
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "rail_track": (230, 150, 140),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "guard_rail": (180, 165, 180),
    "bridge": (150, 100, 100),
    "tunnel": (150, 120, 90),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "caravan": (0, 0, 90),
    "train": (0, 80, 100),
    "license_plate": (0, 0, 142),
}


def hex_to_rgb(hex_value: str):
    """
    Converts a hexadecimal value into a RGB value tuple

    The given hexadecimal value as string is converted into a tuple containing
    the corresponding RGB values as decimals.

    Parameters
    ----------
        hex_value : str
            Hexadecimal value as string with a leading hash sign (#)

    Returns
    -------
    Returns a tuple containing RGB values as decimals.
    """

    hex_value = hex_value.lstrip("#")
    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


# Mapping of body part segmentation RGB values to semantic labels
BODY_PART_SEMANTIC_MAPPING = {
    "eye_left": hex_to_rgb("#00b9ba"),
    "eye_right": hex_to_rgb("#ba0004"),
    "mouth": hex_to_rgb("#7600ba"),
    "nose": hex_to_rgb("#ba00b6"),
    "face": hex_to_rgb("#9efc23"),
    "hair": hex_to_rgb("#dbdbdb"),
    "hat": hex_to_rgb("#aaaaaa"),
    "neck": hex_to_rgb("#f0fc23"),  # Currently not available
    "torso_front": hex_to_rgb("#4623fc"),
    "torso_back": hex_to_rgb("#9423fc"),
    "upper_arm_left_front": hex_to_rgb("#50fc23"),
    "upper_arm_left_back": hex_to_rgb("#50fc23"),
    "upper_arm_right_front": hex_to_rgb("#fc4623"),
    "upper_arm_right_back": hex_to_rgb("#fc4623"),
    "lower_arm_left_front": hex_to_rgb("#23fcb1"),
    "lower_arm_left_back": hex_to_rgb("#23fcb1"),
    "lower_arm_right_front": hex_to_rgb("#fc2383"),
    "lower_arm_right_back": hex_to_rgb("#fc2383"),
    "hand_left": hex_to_rgb("#237bfc"),
    "hand_right": hex_to_rgb("#e123fc"),
    "skirt": hex_to_rgb("#b2da69"),  # Occlusion of legs by dresses or skirts
    "upper_leg_left_front": hex_to_rgb("#a2e174"),
    "upper_leg_left_back": hex_to_rgb("#a2e174"),
    "upper_leg_right_front": hex_to_rgb("#e1d174"),
    "upper_leg_right_back": hex_to_rgb("#e1d174"),
    "lower_leg_left_front": hex_to_rgb("#74e1be"),
    "lower_leg_left_back": hex_to_rgb("#74e1be"),
    "lower_leg_right_front": hex_to_rgb("#e17f74"),
    "lower_leg_right_back": hex_to_rgb("#e17f74"),
    "foot_left": hex_to_rgb("#7495e1"),
    "foot_right": hex_to_rgb("#e174de"),
    "background": hex_to_rgb("#000000"),
}

# Dictionary containing the official dataset split
KIA_DATASET_SPLITS = {
    2: {
        "bit": {
            "train": splits.TRAIN_BIT_TRANCHE_2,
            "val": splits.VAL_BIT_TRANCHE_2,
            "test": splits.TEST_BIT_TRANCHE_2,
        }
    },
    3: {
        "bit": {
            "train": splits.TRAIN_BIT_TRANCHE_3,
            "val": splits.VAL_BIT_TRANCHE_3,
            "test": splits.TEST_BIT_TRANCHE_3,
        }
    },
    4: {
        "bit": {
            "train": splits.TRAIN_BIT_TRANCHE_4,
            "val": splits.VAL_BIT_TRANCHE_4,
            "test": splits.TEST_BIT_TRANCHE_4,
        },
        "mv": {
            "train": splits.TRAIN_MV_TRANCHE_4,
            "val": splits.VAL_MV_TRANCHE_4,
            "test": splits.TEST_MV_TRANCHE_4,
        },
    },
    5: {
        "bit": {
            "train": splits.TRAIN_BIT_TRANCHE_5,
            "val": splits.VAL_BIT_TRANCHE_5,
            "test": splits.TEST_BIT_TRANCHE_5,
        },
        "mv": {
            "train": splits.TRAIN_MV_TRANCHE_5,
            "val": splits.VAL_MV_TRANCHE_5,
            "test": splits.TEST_MV_TRANCHE_5,
        },
    },
    6: {
        "mv": {
            "train": splits.TRAIN_MV_TRANCHE_6,
            "val": splits.VAL_MV_TRANCHE_6,
            "test": splits.TEST_MV_TRANCHE_6,
        }
    },
    7: {
        "mv": {
            "train": splits.TRAIN_MV_TRANCHE_7,
            "val": splits.VAL_MV_TRANCHE_7,
            "test": splits.TEST_MV_TRANCHE_7,
        }
    },
}

# List of the supported dataset tranches
KIA_DATASET_TRANCHES = [2, 3, 4, 5, 6, 7]
