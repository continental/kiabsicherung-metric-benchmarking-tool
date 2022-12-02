# Installation

This document describes how to install this repository. Please follow the instructions step by step.

Make sure you meet the requirements:
* Python 3.7+ (preferably in a virtual environment)
* Ubuntu 18.04 LTS, Ubuntu 20.04 LTS or Red Hat Enterprise Linux 7.3

**WARNING:** This tool is use at own risk. Please inspect the script before running it carefully since it might destroy or overwrite your data!

And for the impatient people:

```bash
git clone https://luxproject.luxoft.com/stash/scm/kia/tp3_ap3.6_e3.6.3_metric_benchmarking_tool.git
cd tp3_ap3.6_e3.6.3_metric_benchmarking_tool
pip install .
```

## 1. Clone repository

Simply clone the repository as you would clone any other repository.
```
git clone https://luxproject.luxoft.com/stash/scm/kia/tp3_ap3.6_e3.6.3_metric_benchmarking_tool.git
```

## 2. (OPTIONAL) Switch Branches

If you want the latest features and bugfixes, you should use the "master" branch. Not that this might be unstable. It is recommended to use a stable release tag. The naming convention is there `vX.X.X`.
If you need a feature, that is currently still in development, checkout a feature branches:

```bash
cd tp3_ap3.6_e3.6.3_metric_benchmarking_tool
git checkout feature/*  # only if you want a feature branch
cd ..
```

The following table shows used branch naming schemes and their meaning.

|     Branch     |                 Description                 |
| :------------: | :-----------------------------------------: |
|     master     | Fairly Stable, contains the latest features |
|   feature/*    |    Unstable, features are developed here    |
|    bugfix/*    |   Unstable, bug fixing in developed here    |
| release/vX.X.X |      A pre-release for version vX.X.X       |

## 3. Update the Repository

Once you are on the branch which you want to use, it is recommended to do a pull first, in case there were any changes. On the initial clone this should not yield updates, but later you can redo the steps to update the repository.

```bash
cd tp3_ap3.6_e3.6.3_metric_benchmarking_tool
git pull
cd ..
```

## 4. Install Tool in the Repository

Now you can install the tool:

```bash
cd tp3_ap3.6_e3.6.3_metric_benchmarking_tool
pip install -e .
cd ..
```

**Note**: You can omit the `-e` flag if you don't want to develop some new
feature and install the MBT into the system.