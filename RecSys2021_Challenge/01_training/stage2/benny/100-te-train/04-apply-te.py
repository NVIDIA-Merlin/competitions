# Copyright 2021 NVIDIA CORPORATION
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
import glob
import pandas as pd
import numpy as np
import cudf
import cupy
import gc

import pickle
import os

from datetime import datetime

from utils import *

import argparse

fold = 0

my_parser = argparse.ArgumentParser(description='NN')
my_parser.add_argument('fold',
                       type=str
                      )

args = my_parser.parse_args()

fold = int(args.fold)
print('fold: ' + str(fold))

TE_files_valid = sorted(glob.glob('/raid/TE_valid/*.parquet'))
CE_files_valid = sorted(glob.glob('/raid/CE_valid/*.parquet'))

os.system('rm -r /raid/recsys2021_pre_validXGB_TE/')
os.system('mkdir /raid/recsys2021_pre_validXGB_TE/')

files = sorted(glob.glob('/raid/recsys2021_valid_pre_split_validXGB/*'))

means_valid = pickle.load(open('/raid/means_valid.pickle', 'rb'))

for file in files:
    print(file)
    add_TE(file, TE_files, TE_files_valid, CE_files_valid, means_valid, fold)