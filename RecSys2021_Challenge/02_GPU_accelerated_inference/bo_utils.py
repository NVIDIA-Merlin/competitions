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
import os
import pandas as pd
import numpy as np
import gc
import datetime
import hashlib
import re
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import BertTokenizer

from datetime import datetime

    
label_names = sorted(['reply', 'retweet', 'retweet_comment', 'like'])
CAT_COLUMNS = ['a_user_id','b_user_id','language','media','tweet_type']
NUMERIC_COLUMNS = ['a_follower_count',
                     'a_following_count',
                     'a_is_verified',
                     'b_follower_count',
                     'b_following_count',
                     'b_is_verified',
                     'b_follows_a',
                     'tw_len_media',
                     'tw_len_photo',
                     'tw_len_video',
                     'tw_len_gif',
                     'tw_len_quest',
                     'tw_len_token',
                     'tw_count_capital_words',
                     'tw_count_excl_quest_marks',
                     'tw_count_special1',
                     'tw_count_hash',
                     'tw_last_quest',
                     'tw_len_retweet',
                     'tw_len_rt',
                     'tw_count_at',
                     'tw_count_words',
                     'tw_count_char',
                     'tw_rt_count_words',
                     'tw_rt_count_char',
                     'len_hashtags',
                     'len_links',
                     'len_domains',
                     'a_ff_rate',
                     'b_ff_rate',
                     'ab_fing_rate',
                     'ab_fer_rate',
                     'a_age',
                     'b_age',
                     'ab_age_dff',
                     'ab_age_rate']

def read_norm_merge(path, split='train'):
    ddf = pd.read_parquet(path)
    ddf['date'] = pd.to_datetime(ddf['timestamp'], unit='s')
    
    if split!='test':
        VALID_DOW = '2021-02-18'
        if split=='train':
            ddf = ddf[ddf['date']<pd.to_datetime(VALID_DOW)].reset_index(drop=True)
        else:
            ddf = ddf[ddf['date']>=pd.to_datetime(VALID_DOW)].reset_index(drop=True)    
    
    ddf['a_ff_rate'] = (ddf['a_following_count'] / ddf['a_follower_count']).astype('float32')
    ddf['b_ff_rate'] = (ddf['b_follower_count']  / ddf['b_following_count']).astype('float32')
    ddf['ab_fing_rate'] = (ddf['a_following_count'] / ddf['b_following_count']).astype('float32')
    ddf['ab_fer_rate'] = (ddf['a_follower_count'] / (1+ddf['b_follower_count'])).astype('float32')
    ddf['a_age'] = ddf['a_account_creation'].astype('int16') + 128
    ddf['b_age'] = ddf['b_account_creation'].astype('int16') + 128
    ddf['ab_age_dff'] = ddf['b_age'] - ddf['a_age']
    ddf['ab_age_rate'] = ddf['a_age']/(1+ddf['b_age'])

    ## Normalize
    for col in NUMERIC_COLUMNS:
        if col == 'tw_len_quest':
            ddf[col] = np.clip(ddf[col].values,0,None)
        if ddf[col].dtype == 'uint16':
            ddf[col].astype('int32')

        if col == 'ab_age_dff':
            ddf[col] = ddf[col] / 256.            
        elif 'int' in str(ddf[col].dtype) or 'float' in str(ddf[col].dtype):    
            ddf[col] = np.log1p(ddf[col])

        if ddf[col].dtype == 'float64':
            ddf[col] = ddf[col].astype('float32') 
            
    ## get categorical embedding id        
    for col in CAT_COLUMNS:
        ddf[col] = ddf[col].astype('float')
        if col in ['a_user_id','b_user_id']:
            mapping_col = 'a_user_id_b_user_id'
        else:
            mapping_col = col
        mapping = pd.read_parquet(f'./categories/unique.{mapping_col}.parquet').reset_index()
        mapping.columns = ['index',col]
        ddf = ddf.merge(mapping, how='left', on=col).drop(columns=[col]).rename(columns={'index':col})
        ddf[col] = ddf[col].fillna(0).astype('int')         

    label_names = ['reply', 'retweet', 'retweet_comment', 'like']
    DONT_USE = ['timestamp','a_account_creation','b_account_creation','engage_time',
                'fold', 'dt_dow', 'a_account_creation', 
                'b_account_creation', 'elapsed_time', 'links','domains','hashtags','id', 'date', 'is_train', 
                'tw_hash0', 'tw_hash1', 'tw_hash2', 'tw_http0', 'tw_uhash', 'tw_hash', 'tw_word0', 
                'tw_word1', 'tw_word2', 'tw_word3', 'tw_word4', 'dt_minute', 'dt_second',
               'dt_day', 'group', 'text', 'tweet_id', 'tw_original_user0', 'tw_original_user1', 'tw_original_user2',
                'tw_rt_user0', 'tw_original_http0', 'tw_tweet',]
    DONT_USE = [c for c in ddf.columns if c in DONT_USE]
    gc.collect(); gc.collect()
    
    return ddf.drop(columns=DONT_USE)