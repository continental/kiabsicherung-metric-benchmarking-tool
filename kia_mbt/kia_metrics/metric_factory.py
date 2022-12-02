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
This file contains a factory class to instantiate available metric processors.

Note that all metrics have assigned an identifier, which is defined in the
metric catalogue. Please have a look at the following Confluence page.
There you will find the document, containing the data catalogue.

https://confluence.vdali.de/x/FQV8Ag
"""

from typing import List
from kia_mbt.kia_metrics.metric_processor import MetricProcessor

# import metric processors
from kia_mbt.kia_metrics.number_of_true_positives import NumberOfTruePositives
from kia_mbt.kia_metrics.number_of_false_negatives import NumberOfFalseNegatives
from kia_mbt.kia_metrics.number_of_false_positives import NumberOfFalsePositives
from kia_mbt.kia_metrics.precision import Precision
from kia_mbt.kia_metrics.recall import Recall
from kia_mbt.kia_metrics.precision_recall_curve import PrecisionRecallCurve
from kia_mbt.kia_metrics.f1score import F1Score
from kia_mbt.kia_metrics.voc_map import VocMAP


class MetricProcessorFactory(object):
    """
    This class provides a factory that instantiates available metric processors.

    Parameters
    ----------
        registration_map : dict
            Map between metric identifiers and the respectiv metric processor
    """

    registration_map: dict = {}

    @classmethod
    def register(cls, identifier: int, processor) -> None:
        """
        Class method that registers a metric processor.

        Parameters
        ----------
            identifier: int
                Identifier of the metric

            processor:
                Class of the respective metric processor
        """

        cls.registration_map.update({identifier: processor})

    @classmethod
    def get_processor(cls, identifier: int) -> MetricProcessor:
        """
        Creates a new instance of a metric processor by identifier.

        Parameters
        ----------
            identifier: int
                Identifier of the metric

        Returns
        -------
        Instance of respective metric processor.
        """

        return cls.registration_map[identifier]()

    @classmethod
    def get_processors(cls, identifiers: List[int]) -> List[MetricProcessor]:
        processors = []
        for identfier in identifiers:
            processors.append(cls.get_processor(identfier))
        return processors

    @classmethod
    def get_name(cls, identifier: int) -> str:
        """
        Get the name of a metric by its identifier.

        Parameters
        ----------
            identifier: int
                Identifier of the metric

        Returns
        -------
        Name of the metric.
        """

        processor = cls.registration_map[identifier]()
        return processor.name

    @classmethod
    def get_all_processors(cls) -> List[MetricProcessor]:
        """
        Get all processors from registration map.

        Returns
        -------
        List with metric processors from the registry.
        """

        processors = []
        for processor in cls.registration_map.values():
            processors.append(processor())
        return processors


# Registration of metric processors
MetricProcessorFactory.register(identifier=1001, processor=F1Score)
MetricProcessorFactory.register(identifier=1003, processor=VocMAP)
MetricProcessorFactory.register(identifier=1027, processor=Precision)
MetricProcessorFactory.register(identifier=1028, processor=Recall)
MetricProcessorFactory.register(identifier=1029, processor=NumberOfTruePositives)
MetricProcessorFactory.register(identifier=1030, processor=NumberOfFalsePositives)
MetricProcessorFactory.register(identifier=1031, processor=NumberOfFalseNegatives)
MetricProcessorFactory.register(identifier=1040, processor=PrecisionRecallCurve)
