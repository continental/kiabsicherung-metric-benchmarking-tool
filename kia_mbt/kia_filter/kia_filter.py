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
Filter module for matched ground truth and prediction data.

"""

from typing import Union, List, Dict
import json
import numpy as np
import pandas as pd
from kia_mbt.config_loader import FilterConfig


class KiaFilter:
    """
    Class to manage filters. The filters are applied to an annotation-, prediction- and a matching data frame.
    Filter operations are primarily regarded as conditioning on data and do not directly influence the matching.

    Attributes
    ----------
        annotation_data : DataFrame
            Ground-truth annotations.

        prediction_data : Dataframe
            Predictions for the ground truth data.

        matching_data : Dataframe
            Matching between annotation_data and prediction_data

        annotation_filter : List[dict]
            List with filters applied to the ground-truth table. Each filter is a
            dictionary of the form {'filter_info': str, 'filter_list': List[bool]}

        prediction_filter : List[dict]
            List with filters applied to the prediction table. Each filter is a
            dictionary of the form {'filter_info': str, 'filter_list': List[bool]}

        matching_filter : List[dict]
            List with filters applied to the matching table. Each filter is a
            dictionary of the form {'filter_info': str, 'filter_list': list[bool]}
    """

    def __init__(self, annotation_data: pd.DataFrame,
                 prediction_data: pd.DataFrame, matching_data: pd.DataFrame,
                 config: FilterConfig):
        """
        Parameters
        ----------
            annotation_data : DataFrame
                Ground-truth annotations.

            prediction_data : DataFrame
                Predictions based on the ground truth data.

            matching_data : DataFrame
                Matching of ground truth and prediction data.

            config : FilterConfig
                Configuration for the filter module.

        """
        self.annotation_data = annotation_data
        self.prediction_data = prediction_data
        self.matching_data = matching_data

        self.annotation_filter = []
        self.prediction_filter = []
        self.matching_filter = []

        # load annotation filters if specified
        if isinstance(config.annotation_filter, dict):
            # apply annotation filter
            for key in config.annotation_filter.keys():
                if isinstance(config.annotation_filter[key], dict):
                    annotation_filter_dict = config.annotation_filter[key]
                elif isinstance(config.annotation_filter[key], list):
                    annotation_filter_dict = self.list_to_filter_dict(
                        filter_args=config.annotation_filter[key])
                else:
                    raise Exception(
                        f"Annotation filter should be dict or list but is {type(config.annotation_filter[key])}"
                    )

                self.apply_relational_filter(
                    filter_dict=annotation_filter_dict,
                    table_identifier='annotation',
                    filter_info=key)

        # load prediction filters if specified
        if isinstance(config.prediction_filter, dict):
            # apply prediction filter
            for key in config.prediction_filter.keys():
                if isinstance(config.prediction_filter[key], dict):
                    prediction_filter_dict = config.prediction_filter[key]
                elif isinstance(config.prediction_filter[key], list):
                    prediction_filter_dict = self.list_to_filter_dict(
                        filter_args=config.prediction_filter[key])
                else:
                    raise Exception(
                        f"Prediction filter should be dict or list but is {type(config.prediction_filter[key])}"
                    )

                self.apply_relational_filter(
                    filter_dict=prediction_filter_dict,
                    table_identifier='prediction',
                    filter_info=key)

        # load matching filters if specified
        if isinstance(config.matching_filter, dict):
            # apply matching filter
            for key in config.matching_filter.keys():
                if isinstance(config.matching_filter[key], dict):
                    matching_filter_dict = config.matching_filter[key]
                elif isinstance(config.matching_filter[key], list):
                    matching_filter_dict = self.list_to_filter_dict(
                        filter_args=config.matching_filter[key])
                else:
                    raise Exception(
                        f"Matching filter should be dict or list but is {type(config.matching_filter[key])}"
                    )
                self.apply_relational_filter(filter_dict=matching_filter_dict,
                                             table_identifier='matching',
                                             filter_info=key)

    @classmethod
    def from_config(cls, annotation_data: pd.DataFrame,
                    prediction_data: pd.DataFrame, matching_data: pd.DataFrame,
                    config: FilterConfig):
        """
        Creates KIA filter from filter module config.

        Parameters
        ----------
            annotation_data : DataFrame
                Ground-truth annotations.

            prediction_data : DataFrame
                Predictions based on the ground truth data.

            matching_data : DataFrame
                Matching of ground truth and prediction data.

            config : FilterConfig
                Configuration for the filter module.

        Returns
        -------
        KIA filter.

        """
        return cls(annotation_data, prediction_data, matching_data, config)

    @classmethod
    def from_config_file(cls,
                         annotation_data: pd.DataFrame,
                         prediction_data: pd.DataFrame,
                         matching_data: pd.DataFrame,
                         config_path: str = None):
        """
        Creates KIA filter from configuration file.

        Parameters
        ----------
            annotation_data : DataFrame
                Ground-truth annotations.

            prediction_data : DataFrame
                Predictions based on the ground truth data.

            matching_data : DataFrame
                Matching of ground truth and prediction data.

            config_path : str
                Path to a .json file with filter configurations

        Returns
        -------
        KIA filter.

        """
        # if a config path is passed the defined filters will be loaded
        config_data = {}
        if config_path:
            try:
                with open(config_path, 'r') as stream:
                    config_data = json.load(stream)
            except FileNotFoundError as exc:
                print(exc, "Skipping loading of KiaFilter config")

        config = FilterConfig.from_payload(config_data)
        return cls(annotation_data, prediction_data, matching_data, config)

    @staticmethod
    def to_filter_dict(column: str, operator: str, value) -> Dict:
        """
        Create a filter dictionary from the passed key words.

        Parameters
        ----------
            column : str
                Name of the column.

            operator : str
                One of: '==', '!=', '>', '>=', '<', '<=', 'in', 'not_in'.

            value:
                Value to compare with the entries of the specified column.

        Returns
        -------
            (dict): Dictionary with filter parameters.

        """
        return {"column": column, "operator": operator, "value": value}

    @staticmethod
    def list_to_filter_dict(filter_args: List) -> Dict:
        """
        Create a filter dictionary from the passed list.

        Parameters
        ----------
            filter_args : List
                List with 3 values: [<column>, <operator>, <value>].

        Returns
        -------
            (dict): Dictionary with filter parameters.

        """
        return {
            "column": filter_args[0],
            "operator": filter_args[1],
            "value": filter_args[2]
        }

    def apply_relational_filter(self,
                                filter_dict: Union[Dict, List],
                                filter_info: str = "",
                                table_identifier: str = "annotation"):
        """
        Applies simple relational filter to the specified data set.
        Implemented relational operations are:
        '==', '!=', '>', '>=', '<', '<=', 'in', 'not_in'

        Parameters
        ----------
            filter_dict : Union[dict, list]
                Dictionary or list of dictionaries with the parameters for simple relational filter operations.
                Dictionaries must have the keys: 'column', 'operator' and 'value'.

            filter_info : str
                Info string for the applied filter. If empty this string will be generated automatically,
                based on information about the total number of filters already applied.

            table_identifier : str
                String to identify the data table to which the filter should be applied.
                One of: "annotation", "prediction", "matching"

        """
        # set default info if empty
        if filter_info == "":
            filter_info = f"filter_" \
                          f"{len(self.annotation_filter) + len(self.prediction_filter) + len(self.matching_filter)}"
        # recursive call if list of filters is provided
        if isinstance(filter_dict, list) and all(
                isinstance(x, dict) for x in filter_dict):
            for idx, relational_filter in enumerate(filter_dict):
                self.apply_relational_filter(
                    filter_dict=relational_filter,
                    filter_info=f"{filter_info}_{str(idx)}",
                    table_identifier=table_identifier)
        # apply single filter from dict
        elif isinstance(filter_dict, dict):

            if table_identifier == 'annotation':
                data = self.annotation_data
                applied_filters = self.annotation_filter

            elif table_identifier == 'prediction':
                data = self.prediction_data
                applied_filters = self.prediction_filter

            elif table_identifier == 'matching':
                data = self.matching_data
                applied_filters = self.matching_filter

            else:
                raise NameError(
                    f"key word 'table_identifier' has to be one of: 'annotation', 'prediction', "
                    f"'matching'. But '{table_identifier}' was passed.")

            filter_list = self.calc_relational_filter(
                data_frame=data,
                column=filter_dict['column'],
                operator=filter_dict['operator'],
                value=filter_dict['value'])

            applied_filters.append({
                'filter_info': filter_info,
                'filter_list': filter_list
            })
        else:
            raise Exception(
                f"Wrong combination of data types. 'filter_dict' and 'filter_info' both have to be lists or"
                f"'filter_dict' be a dictionary and 'filter_info' a string. Actual types: "
                f"type(filter_dict)={type(filter_dict)} and type(filter_info)={type(filter_info)}"
            )

    @staticmethod
    def calc_relational_filter(data_frame: pd.DataFrame, column: str,
                               operator: str, value) -> List[bool]:
        """
        Calculate a boolean filter list according to the defined relational filter with respect to the
        passed data frame. Only operations defined in this method can be handled in the filter config file.

        Parameters
        ----------
            data_frame : DataFrame
                Data frame to which the filter is applied.

            column : str
                Key of the data_frame table.

            operator : str
                Operator for comparison.

            value:
                Value(s) to compare the data_frame entry with

        Returns
        -------
            (List[bool]): Boolean list to indicate whether a row is filtered.
            'True' means that a row will be kept and 'False' means discarded.

        """
        # address nested attributes.
        # E.g. KI-A data uses the column 'size' to store a list with 'height' and 'width'
        key_splitted = column.split('[')
        if len(key_splitted) > 1:
            column = data_frame[key_splitted[0]]
            for unparsed_idx in key_splitted[1:]:
                column_data = pd.Series(
                    [x[int(unparsed_idx[:-1])] for x in column])
        else:
            column_data = data_frame[column]

        # compute boolean list for operator
        if operator == '==':
            is_filtered = (column_data == value)
        elif operator == '!=':
            is_filtered = (column_data != value)
        elif operator == '<':
            is_filtered = (column_data < value)
        elif operator == '>':
            is_filtered = (column_data > value)
        elif operator == '<=':
            is_filtered = (column_data <= value)
        elif operator == '>=':
            is_filtered = (column_data >= value)
        elif operator == 'in':
            is_filtered = (column_data.isin(value))
        elif operator == 'not_in':
            is_filtered = ~(column_data.isin(value))
        else:
            raise NotImplementedError(
                f'filter operator "{operator}" is not defined.')
        return is_filtered

    def apply_annotation_filter(self,
                                filter_list: List[bool] = None,
                                info: str = ''):
        """
        Apply a custom filter to the ground-truth annotation data.

        Parameters
        ----------
            filter_list : List[bool]
                List with booleans to indicate which data is filtered.
                'True' means that a row will be kept and 'False' means discarded.

            info : str
                Description for the applied filter.

        """
        self.annotation_filter.append({
            'filter_info': info,
            'filter_list': filter_list
        })

    def apply_prediction_filter(self, filter_list: List[bool], info: str = ''):
        """
        Apply a custom filter to the prediction data.

        Parameters
        ----------
            filter_list : List[bool]
                List with booleans to indicate which data is filtered.
                'True' means that a row will be kept and 'False' means discarded.

            info : str
                Description for the applied filter.

        """
        self.prediction_filter.append({
            'filter_info': info,
            'filter_list': filter_list
        })

    def apply_matching_filter(self, filter_list: list, info: str = ''):
        """
        Apply a custom filter to the matching data.

        Parameters
        ----------
            filter_list : List[bool]
                List with booleans to indicate which data is filtered.
                'True' means that a row will be kept and 'False' means discarded
            info : str
                Description for the applied filter.

        """
        self.matching_filter.append({
            'filter_info': info,
            'filter_list': filter_list
        })

    def apply_instance_filter(self,
                              instance_list: list,
                              info: str = "instance isin instance_list"):
        """
        Apply a custom filter to select the instances specified in list.

        Parameters
        ----------
            instance_list : List[str]
                List of instance tokens to keep.
            info : str
                Info string. Defaults to "instance isin instance_list".

        """
        filter_list = list(self.annotation_data.index.isin(instance_list))
        self.apply_annotation_filter(filter_list=filter_list, info=info)

    def get_annotation_view(self) -> pd.DataFrame:
        """
        Get a view of the current filtered ground-truth annotation table.

        Returns
        -------
            (DataFrame): pandas DataFrame with remaining annotation data.

        """
        filter_list: List[bool] = self._summarize_filters(
            filter_dicts=self.annotation_filter,
            len_dataframe=self.annotation_data.shape[0])
        return self.annotation_data[filter_list]

    def get_prediction_view(self) -> pd.DataFrame:
        """
        Get a view of the current filtered prediction table.

        Returns
        -------
            (DataFrame): pandas DataFrame with remaining prediction data.

        """
        filter_list: List[bool] = self._summarize_filters(
            filter_dicts=self.prediction_filter,
            len_dataframe=self.prediction_data.shape[0])
        return self.prediction_data[filter_list]

    def get_matching_view(self) -> pd.DataFrame:
        """
        Get a view of the current filtered matching table.

        Returns
        -------
            (DataFrame): pandas DataFrame with remaining matching data.

        """
        filter_list: List[bool] = self._summarize_filters(
            filter_dicts=self.matching_filter,
            len_dataframe=self.matching_data.shape[0])
        return self.matching_data[filter_list]

    def get_view(self):
        """
        Get a view of the current filtered correlation table taking into account the filters applied
        to the ground-truth annotations and prediction tables.

        Returns
        -------
            (DataFrame): pandas DataFrame with remaining matching data taking into account the filters
            applied to the annotation and prediction tables.

        """
        filtered_ground_truth_indices = \
            [not x for x in self._summarize_filters(self.annotation_filter, self.annotation_data.shape[0])]
        filtered_ground_truth_indices = self.annotation_data.index[
            filtered_ground_truth_indices]
        filter_list_ground_truth: List[bool] = []
        for idx in range(self.matching_data.shape[0]):
            is_idx_filtered = self.matching_data['annotation_index'][
                idx] in filtered_ground_truth_indices
            filter_list_ground_truth.append(not is_idx_filtered)

        filtered_prediction_indices = \
            [not x for x in self._summarize_filters(self.prediction_filter, self.prediction_data.shape[0])]
        filtered_prediction_indices = self.prediction_data.index[
            filtered_prediction_indices]
        filter_list_predictions: List[bool] = []
        for idx in range(self.matching_data.shape[0]):
            is_idx_filtered = self.matching_data['detection_index'][
                idx] in filtered_prediction_indices
            filter_list_predictions.append(not is_idx_filtered)

        filter_list: List[bool] = self._summarize_filters(
            self.matching_filter, self.matching_data.shape[0])
        filter_list = (np.array(filter_list) * np.array(filter_list_ground_truth) * np.array(filter_list_predictions)).\
            tolist()
        return self.matching_data[filter_list]

    @staticmethod
    def _summarize_filters(filter_dicts: List[dict],
                           len_dataframe: int = 0) -> Union[bool, List]:
        """
        Summarizes the filters applied to a single table. A row only remains if it is not filtered
        out by any filter.

        Parameters
        ----------
            filter_dicts: List[dict]
                List of dictionaries including booleans to indicate which data is filtered.

            len_dataframe: int
                Number of rows in the dataframe to summarize filters for.

        Returns
        -------
            (List[bool]): List with boolean entries indicating which rows still remains in the view.

        """
        filter_list = np.full(shape=len_dataframe, fill_value=True, dtype=bool)
        for filter_dict in filter_dicts:
            filter_list *= np.array(filter_dict['filter_list'])

        return list(filter_list)
