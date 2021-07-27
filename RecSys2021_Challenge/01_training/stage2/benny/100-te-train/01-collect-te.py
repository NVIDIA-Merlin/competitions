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

def collect_CE(df):
    for col in CE_cols:
        print(col)
        df_tmp = df[
            col + ['reply']
             ].groupby(col).count()
        df_tmp = df_tmp.reset_index()
        df_tmp.columns = col + ['CE_' + col[0]]
        df_tmp = df_tmp.to_parquet('/raid/CE_valid/' + '_'.join(col) + '.parquet')
    del df_tmp
    gc.collect()

my_parser = argparse.ArgumentParser(description='NN')
my_parser.add_argument('fold',
                       type=str
                      )

args = my_parser.parse_args()

fold = int(args.fold)
print('fold: ' + str(fold))

files = glob.glob('/raid/recsys2021_valid_pre/*')
folds = np.load('../../folds.npy')
folds = cupy.asarray(folds)
df = cudf.read_parquet(files)
df['folds'] = folds

os.system('rm -r /raid/CE_valid')
os.system('mkdir /raid/CE_valid')

collect_CE(df)

train = df[~(df['folds']==fold)]

del df
gc.collect()

train['date'] = cudf.to_datetime(train['timestamp'], unit='s')
train['dt_dow']  = train['date'].dt.weekday
train['dt_hour'] = train['date'].dt.hour
train['dt_minute'] = train['date'].dt.minute

means = train[['reply', 'like', 'retweet', 'retweet_comment']].mean().to_pandas().to_dict()

pickle.dump(means, open('/raid/means_valid.pickle', 'wb'))

os.system('rm -r /raid/TE_valid')
os.system('mkdir /raid/TE_valid')

def collect_TE(train):
    for col in TE_cols:
        print(col)
        df_tmp = train[
            col + ['reply', 'retweet', 'retweet_comment', 'like']
             ].groupby(col).agg({
            'reply': ['sum', 'count'],
            'retweet': ['sum'], 
            'retweet_comment': ['sum'], 
            'like': ['sum']}
        )
        df_tmp = df_tmp.reset_index()
        df_tmp.columns = col + [
            'reply_sum',
            'reply_count',
            'retweet_sum',
            'retweet_comment_sum',
            'like_sum']
        df_tmp = df_tmp[col + [
            'reply_sum',
            'reply_count',
            'retweet_sum',
            'retweet_comment_sum',
            'like_sum'
        ]].to_parquet('/raid/TE_valid/' + '_'.join(col) + '.parquet')
    del df_tmp
    gc.collect()

collect_TE(train)