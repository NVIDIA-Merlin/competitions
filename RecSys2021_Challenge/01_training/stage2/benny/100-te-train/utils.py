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

TE_cols = [
    ['a_user_id'],
    ['b_user_id'],
    ['tweet_id']
]

TE_files = [
    '/raid/TE_submission_opt/b_user_id_tweet_type_language.parquet',
    '/raid/TE_submission_opt/b_user_id_a_user_id.parquet',
    '/raid/TE_submission_opt/a_user_id.parquet',
    '/raid/TE_submission_opt/b_is_verified_tweet_type.parquet',
    '/raid/TE_submission_opt/b_user_id.parquet',
    '/raid/TE_submission_opt/tw_original_user0_tweet_type_language.parquet',
    '/raid/TE_submission_opt/tw_original_user1_tweet_type_language.parquet',
    '/raid/TE_submission_opt/tweet_type.parquet'
]

TE_switch_files = [
    '/raid/TE_submission_opt/a_user_id.parquet',
    '/raid/TE_submission_opt/b_user_id.parquet'
]

CE_files = [
    '/raid/TE_submission_opt/a_user_id.parquet',
    '/raid/TE_submission_opt/b_user_id.parquet'
]

CE_cols = [
    ['b_user_id']
]

DONT_USE = ['a_account_creation','b_account_creation','engage_time',
            'fold', 'a_account_creation', 
            'b_account_creation', 'elapsed_time', 'links','domains','hashtags', 'date', 'is_train', 
            'tw_original_http0', 'tw_original_user0', 'tw_original_user1', 'tw_original_user2',
            'tw_rt_count_char', 'tw_rt_count_words', 'tw_rt_user0', 'tw_tweet', 'tw_word0', 'a_user_id_x', 'b_user_id_x',
            'tw_word1', 'tw_word2', 'tw_word3', 'tw_word4', 'tw_count_hash', 'dt_second', 'dt_day']

means = {}
means['reply'] = 0.02846728456689906
means['like'] = 0.3968895210408169
means['retweet'] = 0.08769760903336701
means['retweet_comment'] = 0.006918407917391091

def add_TE(fn, TE_files, TE_files_valid, CE_files_valid, means_valid, fold, psmooth=20):
    df = cudf.read_parquet(fn)
    df['date'] = cudf.to_datetime(df['timestamp'], unit='s')
    df['dt_dow']  = df['date'].dt.weekday
    df['dt_hour'] = df['date'].dt.hour
    df['dt_minute'] = df['date'].dt.minute
    df['dt_second'] = df['date'].dt.second
    df['is_train'] = (~(df['folds']==fold)).astype(np.int8)
    gc.collect()
    print(df['is_train'].mean())
    # TE normal
    print('TE train')
    for i, file in enumerate(TE_files):
        df_tmp = cudf.read_parquet(file)
        col = [x for x in df_tmp.columns if not('reply' in x or 'retweet' in x or 'like' in x)]
        col_rest = [x for x in df_tmp.columns if x not in col]
        df = df.merge(df_tmp, on=col, how='left')
        for key in means.keys():
            df['TE_' + '_'.join(col) + '_' + key] = (((df[key + '_sum'])+means[key]*psmooth)/(df['reply_count']+psmooth))
            if col[0]=='a_user_id' and key=='like':
                df.loc[df['reply_count']<=1000, 'TE_' + '_'.join(col) + '_' + key] = None
            df['TE_' + '_'.join(col) + '_' + key] = df['TE_' + '_'.join(col) + '_' + key].fillna(np.float32(means[key])).round(3)
        df.drop(col_rest, inplace=True, axis=1)
        gc.collect()
        col = [x for x in df_tmp.columns if not('reply' in x or ('retweet' in x and 'tw_len_retweet' not in x) or '_retweet_comment' in x or 'like' in x)]
        if file in TE_switch_files:
            print(col)
            dfcols = list(df_tmp.columns)
            dfcolsnew = []
            for col in dfcols:
                if col == 'b_user_id':
                    dfcolsnew.append('a_user_id')
                elif col == 'a_user_id':
                    dfcolsnew.append('b_user_id')
                else:
                    dfcolsnew.append(col)
            df_tmp.columns = dfcolsnew
            col = [x for x in df_tmp.columns if not('reply' in x or ('retweet' in x and 'tw_len_retweet' not in x) or '_retweet_comment' in x or 'like' in x)]
            col_rest = [x for x in df_tmp.columns if x not in col]
            df = df.merge(df_tmp, on=col, how='left')
            for key in means.keys():
                df['TE_switch_' + '_'.join(col) + '_' + key] = (((df[key + '_sum'])+means[key]*psmooth)/(df['reply_count']+psmooth))
                df['TE_switch_' + '_'.join(col) + '_' + key] = df['TE_switch_' + '_'.join(col) + '_' + key].fillna(np.float32(means[key])).round(3)
            df.drop(col_rest, inplace=True, axis=1)
        del df_tmp
        gc.collect()
    # TE valid
    print('TE valid')
    for i, file in enumerate(TE_files_valid):
        df_tmp = cudf.read_parquet(file)
        col = [x for x in df_tmp.columns if not('reply' in x or ('retweet' in x and 'tw_len_retweet' not in x) or '_retweet_comment' in x or 'like' in x)]
        col_rest = [x for x in df_tmp.columns if x not in col]
        df = df.merge(df_tmp, on=col, how='left')
        for key in means.keys():
            if df_tmp.shape[0]>1000:
                df['TE_valid_' + '_'.join(col) + '_' + key] = (((df[key + '_sum']-df[key]*df['is_train'])+means_valid[key]*psmooth)/(df['reply_count']-df['is_train']+psmooth))
            else:
                df['TE_valid_' + '_'.join(col) + '_' + key] = (((df[key + '_sum'])+means_valid[key]*psmooth)/(df['reply_count']+psmooth))
            if col[0]=='a_user_id' and key=='like':
                df.loc[df['reply_count']<=1000, 'TE_valid_' + '_'.join(col) + '_' + key] = None
            df['TE_valid_' + '_'.join(col) + '_' + key] = df['TE_valid_' + '_'.join(col) + '_' + key].fillna(np.float32(means_valid[key])).round(3)
        df.drop(col_rest, inplace=True, axis=1)
        col = [x for x in df_tmp.columns if not('reply' in x or ('retweet' in x and 'tw_len_retweet' not in x) or '_retweet_comment' in x or 'like' in x)]
    # NN valid:
    print('CE valid')
    for i, file in enumerate(CE_files_valid):
        df_tmp = cudf.read_parquet(file)
        col = [list(df_tmp.columns)[0]]
        df_tmp.columns = col + ['CE_valid_' + col[0]]
        df = df.merge(df_tmp, on=col, how='left')
        df['CE_valid_' + col[0]] = df['CE_valid_' + col[0]].fillna(1)
    df['a_ff_rate'] = (df['a_following_count'] / (1+df['a_follower_count'])).astype('float32')
    df['b_ff_rate'] = (df['b_follower_count']  / (1+df['b_following_count'])).astype('float32')
    df['ab_fing_rate'] = (df['a_following_count'] / (1+df['b_following_count'])).astype('float32')
    df['ab_fer_rate'] = (df['a_follower_count'] / (1+df['b_follower_count'])).astype('float32')
    df['ab_age_dff'] = (df['a_account_creation']-df['b_account_creation'])
    df['ab_age_rate'] = (df['a_account_creation']+129)/(df['b_account_creation']+129)
    final_cols = [x for x in sorted(list(df.columns)) if x not in DONT_USE]
    df[final_cols].to_parquet( '/raid/recsys2021_pre_validXGB_TE/' + fn.split('/')[-1])

def splitvalid(fn, folds):
    df = cudf.read_parquet(fn)
    gc.collect()
    df['id'] = df.index
    df['folds'] = folds
    df['date'] = cudf.to_datetime(df['timestamp'], unit='s')
    df['dt_dow']  = df['date'].dt.weekday
    for i in range(7):
        df[df['dt_dow']==i].to_parquet( '/raid/recsys2021_valid_pre_split/' + str(i) + '_' + fn.split('/')[-1] )
    del df
    gc.collect()