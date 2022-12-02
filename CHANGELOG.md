# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0] - 14. April 2022

### Added

- Added the following metrics: mAP and PR-curve.
- Optimized processing time.
- Added dry run mode, see program option `--dryrun`.

### Changed

- Default IoU threshold for prediction and annotation correlation was changed from `0.1` to `0.5` to match the original evaluation from the Opel SSD r3 v2.
- Renamed some files and folders to match a unified naming schema.
- Reworked console and file logging output for the user.
- Fixed bug when loading additional meta information from .`general-globally-per-frame-analysis-enriched_json` where the ID from light sources change from integers to strings.
- Fixed MinIO backend, which was not correctly loading files.
- Fixed division by zero in IoU scores.

## [v0.1.0] - 30. March 2022

### Added

- Support of the following metrics (per sample and globally): #TP, #FN, #FP, F1-score, precision and recall.
- Each processing module is configurable via a central configuration file (e.g. selection of sequences that shall be processed).
- Support of the enriched post-processed annotation folders `general-globally-per-frame-analysis-enriched_json` and `2d-bounding-box-enriched_json`.
- Output of metric values compliant to the specified TP1/TP3 output format.
- Possibility to add custom filters for specific evaluations.
- Support of adding new metrics easily.