# Copyright (c) 2022 Continental AG and subsidiaries.
# Copyright (c) 2022 Elektronische Fahrwerksysteme GmbH (www.efs-auto.com).
# Copyright (c) 2022 LZR, Bergische Universität Wuppertal (https://www.lzr.uni-wuppertal.de/de/).
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
Entry file for the metric benchmarking tool

"""

import sys
import argparse
import logging
from pathlib import Path
from kia_mbt.kia_io.types import KIADatasetConfig
from kia_mbt.kia_correlate.box_correlator import BoxCorrelator
from kia_mbt.kia_filter.kia_filter import KiaFilter
from kia_mbt.kia_correlate.matching_reduction import MatchingReduction
from kia_mbt.kia_output_writer.kia_writer import KIAWriter
import kia_mbt.config_loader as mbt_config
import kia_mbt.data_loading as mbt_data_loading
import kia_mbt.metric_processing as mbt_metric_proc


def _parse_args():
    """
    Parses the given program arguments.

    Apart from parsing the program arguments, the function configures the
    logger used for outputs. Also it is checked if the arguments are correct.

    Returns
    -------
    Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Data Metric Processor")
    parser.add_argument(
        "--dryrun", action="store_true", help="Dry-run mode will not write any files."
    )
    parser.add_argument(
        "-l",
        "--logfile",
        type=str,
        help="When a log file is set, then file logging is enabled.",
    )
    parser.add_argument("-c", "--config", type=str, help="Configuration file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--list_metrics", action="store_true", help="List all available metrics"
    )
    arguments = parser.parse_args()

    # file logging configuration
    log_format = "[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s"

    # Configurate logging level in dependency of the verbose flag
    logging_level = logging.INFO
    if arguments.verbose:
        logging_level = logging.DEBUG

    # switch between file and command line logging
    if arguments.logfile:
        try:
            # Test if log file can be created
            Path(arguments.logfile).touch()
            logging.basicConfig(
                filename=arguments.logfile, level=logging_level, format=log_format
            )
        except OSError as e:
            logging.error("Cannot create log file. Error: %s", e)
            sys.exit()
    else:
        # command line logging
        logging.basicConfig(level=logging_level, format=log_format)

    # Set logging level for urllib3
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return arguments


def main() -> None:
    """
    Main routine
    """

    # argument parsing
    args = _parse_args()

    logging.info("### Metric Benchmarking Tool")

    # list metrics if requested
    if args.list_metrics:
        mbt_metric_proc.list_metrics()
        sys.exit(0)

    # Show configuration
    if args.dryrun:
        logging.info("# Dry-run enabled.")
    if args.verbose:
        logging.info("# Verbose mode enabled.")

    # load configuration
    logging.info("# Reading configuration")
    config_loader = None
    if args.config:
        config_loader = mbt_config.ConfigLoader(args.config)
    else:
        logging.error("Configuration file missing. Use option -c.")
        sys.exit(0)

    # create dataset configuration and backend for data loading based on config
    io_config = config_loader.get_io_config()
    dataset_config = KIADatasetConfig(sequence_names=io_config.sequences)

    # load 2d bounding box annotations from kia data into data frame
    logging.info("# Loading annotation data")
    annotation_backend = mbt_data_loading.get_backend(
        io_config, io_config.data_path
    )
    annotation_data, _ = mbt_data_loading.load_2dbb_annotations(
        annotation_backend, dataset_config
    )
    logging.info("# Loaded annotation data")
    if args.verbose:
        print(
            "Successfully loaded annotation data with shape {}".format(
                annotation_data.shape
            )
        )

    # load 2d bounding box predictions into data frame
    logging.info("# Loading prediction data")
    prediction_backend = mbt_data_loading.get_backend(
        io_config, io_config.predictions_path
    )
    prediction_data = mbt_data_loading.load_2dbb_predictions(
        prediction_backend, io_config.results_folder, dataset_config
    )
    logging.info("# Loaded prediction data")
    if args.verbose:
        print(
            "Successfully loaded prediction data with shape {}".format(
                prediction_data.shape
            )
        )

    # correlate bounding-box annotations and predictions
    logging.info("# Performing correlation of annotations and predictions")
    correlate_config = config_loader.get_correlate_config()
    box_correlator = BoxCorrelator(
        threshold=correlate_config.iou_threshold,
        matching_type=correlate_config.matching_type,
        clip_truncated_boxes=correlate_config.clip_truncated_boxes,
        **correlate_config.optional_arguments
    )
    matching_data = box_correlator(
        annotation_data=annotation_data, detection_data=prediction_data
    )
    logging.info("# Performed correlation of annotations and predictions")
    if args.verbose:
        print(
            "Successfully computed matching table with shape {}".format(
                matching_data.shape
            )
        )

    # apply filter according to config
    logging.info("# Applying filters")

    logging.info("Using filter options from config")
    filter_config = config_loader.get_filter_config()
    kia_filter = KiaFilter(
        annotation_data=annotation_data,
        prediction_data=prediction_data,
        matching_data=matching_data,
        config=filter_config,
    )

    matching_filtered = kia_filter.get_view()
    logging.info("# Applied filters")
    if args.verbose:
        print(
            "Successfully filtered matching, resulting shape {}".format(
                matching_filtered.shape
            )
        )

    # reduce to 1-to-1 matching
    logging.info("# Reducing matching from correlation")
    kia_reduction = MatchingReduction()
    matching_reduced = kia_reduction.reduce_to_exclusive(matching=matching_filtered)
    logging.info("# Reduced matching from correlation")
    if args.verbose:
        print(
            "Successfully reduced matching, resulting shape {}".format(
                matching_reduced.shape
            )
        )

    # calculate metrics
    logging.info("# Calculating metrics")
    global_metrics, sample_metrics = mbt_metric_proc.calc_metrics(
        annotation_data=annotation_data,
        prediction_data=prediction_data,
        matching=matching_reduced,
        config=config_loader.get_metric_config(),
    )
    logging.info("# Calculated metrics")
    if args.verbose:
        print("Successfully calculated metrics")

    # write metrics outputs
    logging.info("# Writing results to files")
    writer_config = config_loader.get_writer_config()
    writer = KIAWriter(
        version_fpath=writer_config.version_file, backend_path=writer_config.output_path
    )
    if not args.dryrun:
        writer.write_global_metrics(global_metrics=global_metrics)
        writer.write_per_sample_metrics(sample_metrics=sample_metrics)
    logging.info("# Wrote results to files")

    logging.info("# Done")
    logging.info("###")

    sys.exit(0)


if __name__ == "__main__":
    main()
