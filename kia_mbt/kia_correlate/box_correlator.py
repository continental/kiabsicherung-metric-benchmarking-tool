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
Bounding box correlator.

"""

from typing import Tuple, List, Callable, Union
import numpy as np
import pandas as pd


class BoxCorrelator():
    """
    Box correlator for finding matching between annotated and predicted 2d-bounding-boxes

    An instance of Correlator is a callable object that takes 2d bounding-box annotations
    and 2d bounding-box detections in the form of pandas DataFrames and produces a third
    pandas DataFrame that contains all true positive box matches as well as records of
    false positive detections and false negative annotations (based on the detections).

    Attributes
    ----------
    threshold : float
        Threshold for matching criterion.

    matching_type : str
        "complete" for all overlapping bounding-box matchings (n-to-m matching).
        "exclusive" for one-to-one matching based on confidence score.

    confidence_col : str
        Name of the confidence value column.
    annotation_bb_center_col : str
        Name of annotation bounding-box center column.
    annotation_bb_size_col : str
        Name of annotation bounding-box size column.
    detection_bb_center_col : str
        Name of detection bounding-box center column.
    detection_bb_size_col : str
        Name of detection bounding-box size column.
    clip_x : Tuple[float, float]
        Coordinates for clip along x-Axis (min. and max. value).
    clip_y : Tuple[float float]
        Coordinates for clip along y-Axis (min. and max. value).

    """

    def __init__(self,
                 threshold: float = 0.5,
                 matching_type: str = "complete",
                 clip_truncated_boxes: bool = True,
                 **kwargs):
        """
        Setup of the Correlator.

        Parameters
        ----------
        threshold : float
            Threshold for matching criterion.
        matching_type : str
            "complete" for all overlapping bounding-box matchings (n-to-m matching).
            "exclusive" for one-to-one matching based on confidence score.
        clip_truncated_boxes : bool
            Clip coordinates of truncated bounding-boxes in IOU (matching_value) computation.

        Kwargs
        ------
        confidence_col : str
            Name of confidence value column.
            Defaults to "confidence".
        annotation_bb_center_col : str
            Name of annotation bounding-box center column.
            Defaults to "center".
        annotation_bb_size_col : str
            Name of annotation bounding-box size column.
            Defaults to "size".
        detection_bb_center_col : str
            Name of detection bounding-box center column.
            Defaults to "center".
        detection_bb_size_col : str
            Name of detection bounding-box size column.
            Defaults to "size".
        clip_x : Tuple[float, float]
            Coordinates for clipping along x-Axis (min. and max. value).
            Defaults to (0.0, 1920.0) if truncate_boxes is True, (-np.inf, np.inf) otherwise.
        clip_y : Tuple[float, float]
            Coordinates for clipping along y-Axis (min. and max. value).
            Defaults to (0.0, 1280.0) if truncate_boxes is True, (-np.inf, np.inf) otherwise.

        """
        self._threshold = threshold
        self._matching_type = matching_type
        self._clip_truncated_boxes = clip_truncated_boxes

        # additional keyword arguments
        self._confidence_col = kwargs.get("confidence_col", "confidence")

        self._annotation_bb_center_col = kwargs.get("annotation_bb_center_col", "center")
        self._annotation_bb_size_col = kwargs.get("annotation_bb_size_col", "size")

        self._detection_bb_center_col = kwargs.get("detection_bb_center_col", "center")
        self._detection_bb_size_col = kwargs.get("detection_bb_size_col", "size")

        # default values for kia data if clip_x is not set
        self._clip_x = kwargs.get("clip_x", None)
        if self._clip_truncated_boxes:
            if self._clip_x is None:
                self._clip_x = (0.0, 1920.0)
        else:
            self._clip_x = (-np.inf, np.inf)

        # default values for kia data if clip_y is not set
        self._clip_y = kwargs.get("clip_y", None)
        if self._clip_truncated_boxes:
            if self._clip_y is None:
                self._clip_y = (0.0, 1280.0)
        else:
            self._clip_y = (-np.inf, np.inf)

        # select appropriate match_boxes implementation
        if self._matching_type == "complete":
            self._match_boxes = self._match_boxes_complete
        elif self._matching_type == "exclusive":
            self._match_boxes = self._match_boxes_exclusive
        else:
            raise RuntimeError("Unknown matching_type in Correlator encountered.")

    def __call__(self,
                 annotation_data: pd.DataFrame,
                 detection_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make the Correlator a callable object for easy use.

        Parameters
        ----------
        annotation_data : pandas.DataFrame
            Ground-truth annotations.
        detection_data : pandas.DataFrame
            Predicted SSD detections.

        Returns
        -------
        matching: pandas.DataFrame
            Data frame containing bounding-box matching.

        """
        criterion_kwargs = dict()
        criterion_kwargs["clip_x"] = self._clip_x
        criterion_kwargs["clip_y"] = self._clip_y

        matching = self.match(annotation_data=annotation_data,
                              detection_data=detection_data,
                              criterion=self._compute_iou,
                              criterion_kwargs=criterion_kwargs,
                              threshold=self._threshold,
                              match_classes=None,
                              annotation_bb_center_col=self._annotation_bb_center_col,
                              annotation_bb_size_col=self._annotation_bb_size_col,
                              detection_bb_center_col=self._detection_bb_center_col,
                              detection_bb_size_col=self._detection_bb_size_col)
        return matching

    def _convert_coords(self,
                        center: Tuple[int, int],
                        size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """
        Convert bounding-box coordinates from center-size to min-max.

        Coordinates are defined such that the vertial (y-Axis) is pointing downwards.

        :  (x_min, y_min)--------------+
        :        |                     |
        :      height   (c_x, c_y)     |
        :        |                     |
        :        +---- width ----(x_max, y_max)

        Parameters
        ----------
        center : Tuple[int, int]
            Center coordinates of bounding-box.
        size : Tuple[int, int]
            Width and height of bounding-box.

        Returns
        -------
        (x_min, y_min, x_max, y_max): Tuple[float, float, float, float]
            Min-max coordinates of the bounding-box.

        """
        # unpack center and size
        c_x, c_y = center
        width, height = size

        # upper left corner
        x_min = float(c_x) - 0.5 * float(width)
        y_min = float(c_y) - 0.5 * float(height)

        # lower right corner
        x_max = float(c_x) + 0.5 * float(width)
        y_max = float(c_y) + 0.5 * float(height)

        return x_min, y_min, x_max, y_max

    def _clip_coords(self,
                     x_min: float,
                     y_min: float,
                     x_max: float,
                     y_max: float,
                     clip_x_min: float = -np.inf,
                     clip_y_min: float = -np.inf,
                     clip_x_max: float = np.inf,
                     clip_y_max: float = np.inf) -> Tuple[float, float, float, float]:
        """
        Truncate coordinates

        Parameters
        ----------
        x_min : float
            x-coordinate of upper left corner.
        y_min : float
            y-coordinate of upper left corner.
        x_max : float
            x-coordinate of lower right corner.
        y_max : float
            y-coordinate of lower right corner.
        clip_x_min : float
            Minimal x-coordinate of the frame.
        clip_y_min : float
            Minimal y-coordinate of the frame
        clip_x_max : float
            Maximum x-coordinate of the frame.
        clip_y_max : float
            Maximum y-coordinate of the frame.

        Returns
        -------
        (x_min, y_min, x_max, y_max): Tuple[float, float, float, float]
            Clipped bounding-box coordinates.

        """
        # make sure (x_max, y_max) >= (clip_x_min, clip_y_min)
        # necessary if the entire box is shifted to the left
        x_max = max(x_max, clip_x_min)
        # necessary if the entire box is shifted to the top
        y_max = max(y_max, clip_y_min)

        # make sure (x_max, y_max) <= (clip_x_max, clip_y_max)
        x_max = min(x_max, clip_x_max)
        y_max = min(y_max, clip_y_max)

        # make sure (x_min, y_min) <= (clip_x_max, clip_y_max)
        # necessary if the entire box is shifted to the right
        x_min = min(x_min, clip_x_max)
        # necessary if the entire box is shifted to the bottom
        y_min = min(y_min, clip_y_max)

        # make sure (x_min, y_min) >= (clip_x_min, clip_y_min)
        x_min = max(x_min, clip_x_min)
        y_min = max(y_min, clip_y_min)

        return x_min, y_min, x_max, y_max

    def _compute_iou(self,
                     center1: Tuple[int, int],
                     size1: Tuple[int, int],
                     center2: Tuple[int, int],
                     size2: Tuple[int, int],
                     **kwargs) -> float:
        """
        Compute the intersection over union (IOU) of two bounding-boxes.

        Parameters
        ----------
        center1 : Tuple[int, int]
            Center coordinates of first box.
        size1 : Tuple[int, int]
            Width and height of first box.
        center2 : Tuple[int, int]
            Center coordinates of second box.
        size2 : Tuple[int, int]
            Width and height of second box.

        Kwargs
        ------
        clip_x : Tuple[float, float]
            Tuple specifying min. and max. x-coordinate for clipping.
        clip_y : Tuple[float, float]
            Tuple specifying min. and max. y-coordinate for clipping.
        eps : float
            Epsilon to avoid zero division.

        Returns
        -------
        iou : float
            Intersection over union score.

        """
        # extract kwargs
        clip_x: Tuple[float, float] = kwargs.get("clip_x", (-np.inf, np.inf))
        clip_y: Tuple[float, float] = kwargs.get("clip_y", (-np.inf, np.inf))
        eps: float = kwargs.get("eps", 1e-12)

        # convert center/size to min-max coordinates
        x_min1, y_min1, x_max1, y_max1 = self._convert_coords(center=center1,
                                                              size=size1)
        x_min2, y_min2, x_max2, y_max2 = self._convert_coords(center=center2,
                                                              size=size2)

        # clip bounding-box coordinates accordingly
        clip_x_min, clip_x_max = clip_x
        clip_y_min, clip_y_max = clip_y

        # clip box 1 coordinates
        x_min1, y_min1, x_max1, y_max1 = self._clip_coords(x_min=x_min1,
                                                           y_min=y_min1,
                                                           x_max=x_max1,
                                                           y_max=y_max1,
                                                           clip_x_min=clip_x_min,
                                                           clip_y_min=clip_y_min,
                                                           clip_x_max=clip_x_max,
                                                           clip_y_max=clip_y_max)

        # clip box 2 coordinates
        x_min2, y_min2, x_max2, y_max2 = self._clip_coords(x_min=x_min2,
                                                           y_min=y_min2,
                                                           x_max=x_max2,
                                                           y_max=y_max2,
                                                           clip_x_min=clip_x_min,
                                                           clip_y_min=clip_y_min,
                                                           clip_x_max=clip_x_max,
                                                           clip_y_max=clip_y_max)

        # compute (x, y) - coordinates of intersection
        x_min = max(x_min1, x_min2)
        x_max = min(x_max1, x_max2)

        y_min = max(y_min1, y_min2)
        y_max = min(y_max1, y_max2)

        # compute area of intersection
        inter_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

        # compute areas of inididual bounding-boxes
        box_area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
        box_area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

        # compute intersection over union
        iou = inter_area / (box_area1 + box_area2 - inter_area + eps)
        return iou

    def _filter_for_sample_name(self,
                                data_frame: pd.DataFrame,
                                sample_name: str) -> pd.DataFrame:
        """
        Filter dataframe for a single sample name.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            Input dataframe to filter.
        sample_name : str
            Sample name to filter for.

        Returns
        -------
        sample_data: pandas.DataFrame
            Data frame containing just single sample.

        """
        sample_data = data_frame[data_frame["sample_name"] == sample_name]
        return sample_data

    def _filter_for_class_id(self,
                             data_frame: pd.DataFrame,
                             class_id: str) -> pd.DataFrame:
        """
        Filter large dataframe for single class id.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            Input data frame to filter.
        class_id : str
            Class index to filter for.

        Returns
        -------
        sample_data: pandas.DataFrame
            Data frame containing just single class.

        """
        class_data = data_frame[data_frame["class_id"] == class_id]
        return class_data

    def _build_matching_entry(self,
                              sample_name: str,
                              annotation_index: str,
                              detection_index: str,
                              confusion: str,
                              class_id: str,
                              match_value: float,
                              confidence: float) -> dict:
        """
        Build a matching entry dict.

        Parameters
        ----------
        sample_name : str

        annotation_index : str
            Annotation index.
        detection_index : str
            Detection index.
        confusion : str
            One of:
                tp: True postive
                fp: False positive
                fn: False negative
        class_id : str
            Class id or np.nan
        match_value : float
            Matching (i.e. IOU) value.
        confidence : float
            Confidence value of prediction or np.nan.

        Returns
        -------
        match_entry: dict
            Entry dict for matching list.

        """
        match_entry = dict()

        match_entry["sample_name"] = sample_name
        match_entry["annotation_index"] = annotation_index
        match_entry["detection_index"] = detection_index
        match_entry["confusion"] = confusion
        match_entry["class_id"] = class_id
        match_entry["match_value"] = match_value
        match_entry["confidence"] = confidence

        return match_entry

    def match(self,
              annotation_data: pd.DataFrame,
              detection_data: pd.DataFrame,
              criterion: Callable[[Tuple, Tuple, Tuple, Tuple, dict], float],
              criterion_kwargs: dict,
              threshold: float,
              match_classes: Union[List[str], None] = None,
              confidence_col: str = "confidence",
              annotation_bb_center_col: str = "center",
              annotation_bb_size_col: str = "size",
              detection_bb_center_col: str = "center",
              detection_bb_size_col: str = "size") -> pd.DataFrame:
        """
        Match bounding-boxes between ground-truth annotations and algorithm detections.

        Parameters
        ----------
        annotation_data : pandas.DataFrame
            Ground-truth annotations.
        detection_data : pandas.DataFrame
            Predicted SSD detections.
        match_classes : List[str]:
            List of classes to include in matching. None includes all classes.
        criterion : Callable
            Callable criterion to compute overlap of bounding-boxes.
        criterion_kwargs: dict
            Dictionary to pass as kwargs to criterion callable.
        threshold : float
            Threshold for matching criterion.

        Returns
        -------
        matching: pandas.DataFrame
            Data frame containing bounding-box matching.

        """
        # initialize matching dict
        matching = list()

        # get all sample names from ground-truth anotations
        sample_name_list = sorted(list(annotation_data["sample_name"].unique()))

        # get all class ids in annotation and detection
        annotation_class_ids = set(annotation_data["class_id"].unique())
        detection_class_ids = set(detection_data["class_id"].unique())
        class_ids = annotation_class_ids.union(detection_class_ids)

        # match only specific classes if match_classes is not None
        if match_classes:
            class_ids = class_ids.intersection(set(match_classes))

        class_id_list = sorted(list(class_ids))

        for sample_name in sample_name_list:
            # filter for sample_name
            annotations_sample = self._filter_for_sample_name(data_frame=annotation_data,
                                                              sample_name=sample_name)

            detections_sample = self._filter_for_sample_name(data_frame=detection_data,
                                                             sample_name=sample_name)

            for class_id in class_id_list:
                # filter for class_id
                annotations_class = self._filter_for_class_id(data_frame=annotations_sample,
                                                              class_id=class_id)

                detections_class = self._filter_for_class_id(data_frame=detections_sample,
                                                             class_id=class_id)

                # determine true positive box matching
                tp_matches, matched_annotation_ids, matched_detection_ids = self._match_boxes(
                                                                    annotations=annotations_class,
                                                                    detections=detections_class,
                                                                    criterion=criterion,
                                                                    criterion_kwargs=criterion_kwargs,
                                                                    threshold=threshold,
                                                                    confidence_col=confidence_col,
                                                                    annotation_bb_center_col=annotation_bb_center_col,
                                                                    annotation_bb_size_col=annotation_bb_size_col,
                                                                    detection_bb_center_col=detection_bb_center_col,
                                                                    detection_bb_size_col=detection_bb_size_col)
                matching = matching + tp_matches

                # determine false positives (unmatched detections)
                fp_detection_ids = set(detections_class.index) - set(matched_detection_ids)
                for fp_det_id in fp_detection_ids:
                    match_entry = self._build_matching_entry(sample_name=sample_name,
                                                             annotation_index=None,
                                                             detection_index=fp_det_id,
                                                             confusion="fp",
                                                             class_id=class_id,
                                                             match_value=np.nan,
                                                             confidence=detections_class.loc[fp_det_id][confidence_col])
                    matching.append(match_entry)

                # determine false negatives (unmatched annotations)
                fn_annotation_ids = set(annotations_class.index) - set(matched_annotation_ids)
                for fn_ann_id in fn_annotation_ids:
                    match_entry = self._build_matching_entry(sample_name=sample_name,
                                                             annotation_index=fn_ann_id,
                                                             detection_index=None,
                                                             confusion="fn",
                                                             class_id=class_id,
                                                             match_value=np.nan,
                                                             confidence=np.nan)
                    matching.append(match_entry)

        # cast to dataframe and return
        matching = pd.DataFrame(matching,
                                columns=["sample_name",
                                         "annotation_index",
                                         "detection_index",
                                         "confusion",
                                         "class_id",
                                         "match_value",
                                         confidence_col])
        return matching

    def _match_boxes_complete(self,
                              annotations: pd.DataFrame,
                              detections: pd.DataFrame,
                              criterion: Callable[[Tuple, Tuple, Tuple, Tuple, dict], float],
                              criterion_kwargs: dict,
                              threshold: float,
                              confidence_col: str,
                              annotation_bb_center_col: str,
                              annotation_bb_size_col: str,
                              detection_bb_center_col: str,
                              detection_bb_size_col: str) -> Tuple[List[dict], List[str], List[str]]:
        """Complete matching of detection and annotation bounding-boxes for single sample and class.
        Assumes, detections and annotations contain only one unique sample_name and class_id each.

        Parameters
        ----------
        annotations : pandas.DataFrame
            Ground-truth annotations.
        detections : pandas.DataFrame
            Predicted SSD detections.
        criterion : Callable
            Callable criterion to compute matching of bounding-boxes.
        criterion_kwargs : dict
            Dictionary to pass as kwargs to criterion callable.
        threshold : float
            Threshold for matching criterion.

        Returns
        -------
        tp_matching : List[dict]
            List of true positive matchings.
        matched_annotation_ids : List[str]
            List of matched annotation indices.
        matched_detection_ids : List[str]
            List of matched detections indices.

        """
        tp_matching = list()
        matched_annotation_ids = set()
        matched_detection_ids = set()

        # iterate over detected boxes
        for det_id, det in detections.iterrows():
            detection_center = det[detection_bb_center_col]
            detection_size = det[detection_bb_size_col]

            # iterate over ground-truth boxes
            for ann_id, ann in annotations.iterrows():
                annotation_center = ann[annotation_bb_center_col]
                annotation_size = ann[annotation_bb_size_col]

                # compute value of matching criterion
                match_value = criterion(detection_center,
                                        detection_size,
                                        annotation_center,
                                        annotation_size,
                                        **criterion_kwargs)

                # build true positive matching and note matched ids
                if match_value >= threshold:
                    match_entry = self._build_matching_entry(sample_name=ann["sample_name"],
                                                             annotation_index=ann_id,
                                                             detection_index=det_id,
                                                             confusion="tp",
                                                             class_id=ann["class_id"],
                                                             match_value=match_value,
                                                             confidence=det[confidence_col])
                    tp_matching.append(match_entry)
                    matched_annotation_ids.add(ann_id)
                    matched_detection_ids.add(det_id)

        return tp_matching, list(matched_annotation_ids), list(matched_detection_ids)

    def _match_boxes_exclusive(self,
                               annotations: pd.DataFrame,
                               detections: pd.DataFrame,
                               criterion: Callable[[Tuple, Tuple, Tuple, Tuple, dict], float],
                               criterion_kwargs: dict,
                               threshold: float,
                               confidence_col: str,
                               annotation_bb_center_col: str,
                               annotation_bb_size_col: str,
                               detection_bb_center_col: str,
                               detection_bb_size_col: str) -> Tuple[List[dict], List[str], List[str]]:
        """Exclusive matching of detection and annotation bounding-boxes in single sample and class.
        Assumes, detections and annotations contain only one unique sample_name and class_id each.

        Parameters
        ----------
        annotations: pandas.DataFrame
            Ground-truth annotations.
        detections: pandas.DataFrame
            Predicted SSD detections.
        match_classes: List[str]:
            List of classes to include in matching. None includes all classes.
        criterion: Callable
            Callable criterion to compute matching of bounding-boxes.
        criterion_kwargs : dict
            Dictionary to pass as kwargs to criterion callable.
        threshold: float
            Threshold for matching criterion.

        Returns
        -------
        tp_matching : List[dict]
            List of true positive matchings.
        matched_annotation_ids : List[str]
            List of matched annotation indices.
        matched_detection_ids : List[str]
            List of matched detections indices.

        """
        tp_matching = list()
        matched_annotation_ids = set()
        matched_detection_ids = set()

        # sort the detected bounding-boxes by confidence column
        sorted_detections = detections.sort_values(by=confidence_col,
                                                   ascending=False)

        # iterate over detected boxes
        for det_id, det in sorted_detections.iterrows():
            detection_center = det[detection_bb_center_col]
            detection_size = det[detection_bb_size_col]

            # current matching value and annotation
            max_match_value = -np.inf
            max_ann_id = None

            # iterate over ground-truth boxes
            for ann_id, ann in annotations.iterrows():
                annotation_center = ann[annotation_bb_center_col]
                annotation_size = ann[annotation_bb_size_col]

                # compute value of matching criterion
                match_value = criterion(detection_center,
                                        detection_size,
                                        annotation_center,
                                        annotation_size,
                                        **criterion_kwargs)

                if match_value > max_match_value:
                    max_ann_id = ann_id
                    max_match_value = match_value

            # check if largest overlap is above threshold
            if max_match_value >= threshold:
                # check if annotation was not yet matched with higher confidence before
                if not max_ann_id in matched_annotation_ids:
                    match_entry = self._build_matching_entry(sample_name=ann["sample_name"],
                                                             detection_index=det_id,
                                                             annotation_index=max_ann_id,
                                                             confusion="tp",
                                                             class_id=annotations.loc[max_ann_id]["class_id"],
                                                             match_value=max_match_value,
                                                             confidence=det[confidence_col])
                    tp_matching.append(match_entry)
                    matched_annotation_ids.add(max_ann_id)
                    matched_detection_ids.add(det_id)

        return tp_matching, list(matched_annotation_ids), list(matched_detection_ids)
