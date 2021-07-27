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

path = '/raid/results/'

pred_files = [
    'results_benny_xgb.csv',
    'results_benny_nn.csv',
    'results_bo.csv',
    'results-chris-nn.csv',
    'results-chris-xgb.csv',
    'results-giba-xgb.csv'
]

colnames = [
    ['tweet_id_org','b_user_id_org','XGB_reply','XGB_retweet','XGB_retweet_comment','XGB_like'],
    ['tweet_id_org','b_user_id_org','BNN_reply','BNN_retweet','BNN_retweet_comment','BNN_like'],
    ['tweet_id_org','b_user_id_org','BONN_reply','BONN_retweet','BONN_retweet_comment','BONN_like'],
    ['tweet_id_org','b_user_id_org','CNN_reply','CNN_retweet','CNN_retweet_comment','CNN_like'],
    ['tweet_id_org','b_user_id_org','CXGB_reply','CXGB_retweet','CXGB_retweet_comment','CXGB_like'],
    ['tweet_id_org','b_user_id_org','GXGB_reply','GXGB_retweet','GXGB_retweet_comment','GXGB_like']
]

my_parser = argparse.ArgumentParser(description='NN')
my_parser.add_argument('fold',
                       type=str
                      )

args = my_parser.parse_args()

fold = int(args.fold)
print('fold: ' + str(fold))

files = glob.glob('/raid/recsys2021_valid_pre_split/*')
os.system('rm -r /raid/recsys2021_valid_pre_split_validXGB')
os.system('mkdir /raid/recsys2021_valid_pre_split_validXGB')

for file in files:
    print(file)
    df = cudf.read_parquet(file)
    #df = pd.read_parquet(file)
    print(df.shape)
    for i, pred_file in enumerate(pred_files):
        print(pred_file)
        sub0 = pd.read_csv(path + pred_file, 
                   header=None, dtype={0:object,1:object,
                                       2:np.float32,3:np.float32,
                                       4:np.float32,5:np.float32}
                  )
        sub0.columns = colnames[i]
        sub0['tweet_id'] = sub0['tweet_id_org'].apply(lambda x: int(x[-16:],16) ).astype(np.int64)
        sub0['b_user_id'] = sub0['b_user_id_org'].apply(lambda x: int(x[-16:],16) ).astype(np.int64)
        sub0.drop(['tweet_id_org', 'b_user_id_org'], inplace=True, axis=1)
        subcudf = cudf.from_pandas(sub0)
        df = df.merge(subcudf, how='left', on=['b_user_id','tweet_id'])
    print(df.shape)
    print(df.isna().sum())
    df.to_parquet( '/raid/recsys2021_valid_pre_split_validXGB/' + file.split('/')[-1])
