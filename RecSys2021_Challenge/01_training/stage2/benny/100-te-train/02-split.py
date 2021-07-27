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

os.system('rm -r /raid/recsys2021_valid_pre_split/')
os.system('mkdir /raid/recsys2021_valid_pre_split/')

folds = np.load('../../folds.npy')
folds = cupy.asarray(folds)

files = sorted(glob.glob('/raid/recsys2021_valid_pre/*'))

for file in files:
    print(file)
    splitvalid(file, folds)
