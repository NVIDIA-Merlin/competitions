#
# The MIT License (MIT)

# Copyright (c) 2021, NVIDIA CORPORATION

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import glob
import itertools
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, NamedTuple

import torch

logger = logging.getLogger(__name__)


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def get_filenames(data_paths, files_filter_pattern="*"):
    paths = [
        [p for p in glob.glob(os.path.join(path, files_filter_pattern))]
        for path in data_paths
    ]
    return list(itertools.chain.from_iterable(paths))


def get_label_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    label_feature_config = list(
        [k for k, v in feature_map.items() if "is_label" in v and v["is_label"]]
    )

    if len(label_feature_config) == 0:
        raise ValueError("One feature have be configured as label (is_label = True)")
    if len(label_feature_config) > 1:
        raise ValueError("Only one feature can be selected as label (is_label = True)")
    label_name = label_feature_config[0]
    return label_name


def get_timestamp_feature_name(feature_map: Dict[str, Any]) -> str:
    """
        Analyses the feature map config and returns the name of the label feature (e.g. item_id)
    """
    timestamp_feature_name = list(
        [k for k, v in feature_map.items() if v["dtype"] == "timestamp"]
    )

    if len(timestamp_feature_name) == 0:
        raise Exception(
            'No feature have be configured as timestamp (dtype = "timestamp")'
        )
    if len(timestamp_feature_name) > 1:
        raise Exception(
            'Only one feature can be configured as timestamp (dtype = "timestamp")'
        )

    timestamp_fname = timestamp_feature_name[0]
    return timestamp_fname


def get_parquet_files_names(data_args, time_indices, is_train, eval_on_test_set=False):
    if type(time_indices) is not list:
        time_indices = [time_indices]

    time_window_folders = [
        os.path.join(
            data_args.data_path,
            str(time_index).zfill(data_args.time_window_folder_pad_digits),
        )
        for time_index in time_indices
    ]
    if is_train:
        data_filename = "train.parquet"
    else:
        if eval_on_test_set:
            data_filename = "test.parquet"
        else:
            data_filename = "valid.parquet"

    parquet_paths = [
        os.path.join(folder, data_filename) for folder in time_window_folders
    ]

    ##If paths are folders, list the parquet file within the folders
    # parquet_paths = get_filenames(parquet_paths, files_filter_pattern="*.parquet"

    return parquet_paths

