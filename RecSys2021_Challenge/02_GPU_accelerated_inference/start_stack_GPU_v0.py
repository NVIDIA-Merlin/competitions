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

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import pickle
import gc
import pandas as pd
import numpy as np

import xgboost as xgb

import time
import cudf
import cupy

from cuml import ForestInference

train_features_stage2 = {'reply': ['TE_a_user_id_like',
  'TE_a_user_id_reply',
  'TE_a_user_id_retweet',
  'TE_a_user_id_retweet_comment',
  'TE_b_is_verified_tweet_type_like',
  'TE_b_is_verified_tweet_type_reply',
  'TE_b_is_verified_tweet_type_retweet',
  'TE_b_is_verified_tweet_type_retweet_comment',
  'TE_b_user_id_a_user_id_like',
  'TE_b_user_id_a_user_id_reply',
  'TE_b_user_id_a_user_id_retweet',
  'TE_b_user_id_a_user_id_retweet_comment',
  'TE_b_user_id_like',
  'TE_b_user_id_reply',
  'TE_b_user_id_retweet',
  'TE_b_user_id_retweet_comment',
  'TE_b_user_id_tweet_type_language_like',
  'TE_b_user_id_tweet_type_language_reply',
  'TE_b_user_id_tweet_type_language_retweet',
  'TE_b_user_id_tweet_type_language_retweet_comment',
  'TE_tw_original_user0_tweet_type_language_like',
  'TE_tw_original_user0_tweet_type_language_reply',
  'TE_tw_original_user0_tweet_type_language_retweet',
  'TE_tw_original_user0_tweet_type_language_retweet_comment',
  'TE_tw_original_user1_tweet_type_language_like',
  'TE_tw_original_user1_tweet_type_language_reply',
  'TE_tw_original_user1_tweet_type_language_retweet',
  'TE_tw_original_user1_tweet_type_language_retweet_comment',
  'TE_tweet_type_like',
  'TE_tweet_type_reply',
  'TE_tweet_type_retweet',
  'TE_tweet_type_retweet_comment',
  'a_ff_rate',
  'a_follower_count',
  'a_following_count',
  'a_is_verified',
  'ab_age_dff',
  'ab_age_rate',
  'ab_fer_rate',
  'ab_fing_rate',
  'b_ff_rate',
  'b_follower_count',
  'b_following_count',
  'b_follows_a',
  'b_is_verified',
  'dt_dow',
  'dt_hour',
  'dt_minute',
  'language',
  'len_domains',
  'len_hashtags',
  'len_links',
  'media',
  'tw_count_at',
  'tw_count_capital_words',
  'tw_count_char',
  'tw_count_excl_quest_marks',
  'tw_count_special1',
  'tw_count_words',
  'tw_last_quest',
  'tw_len_gif',
  'tw_len_media',
  'tw_len_photo',
  'tw_len_quest',
  'tw_len_retweet',
  'tw_len_rt',
  'tw_len_token',
  'tw_len_video',
  'tweet_type',
  'CNN_reply',
  'CNN_retweet',
  'CNN_retweet_comment',
  'CNN_like',
  'CXGB_reply',
  'CXGB_retweet',
  'CXGB_retweet_comment',
  'CXGB_like',
  'BONN_reply',
  'BONN_retweet',
  'BONN_retweet_comment',
  'BONN_like',
  'XGB_reply',
  'XGB_retweet',
  'XGB_retweet_comment',
  'XGB_like',
  'BNN_reply',
  'BNN_retweet',
  'BNN_retweet_comment',
  'BNN_like',
  'GXGB_reply',
  'GXGB_retweet',
  'GXGB_retweet_comment',
  'GXGB_like',
  'TE_valid_a_user_id_like',
  'TE_valid_a_user_id_reply',
  'TE_valid_a_user_id_retweet',
  'TE_valid_a_user_id_retweet_comment',
  'TE_valid_b_user_id_like',
  'TE_valid_b_user_id_reply',
  'TE_valid_b_user_id_retweet',
  'TE_valid_b_user_id_retweet_comment',
  'TE_switch_a_user_id_like',
  'TE_switch_a_user_id_reply',
  'TE_switch_a_user_id_retweet',
  'TE_switch_a_user_id_retweet_comment',
  'TE_switch_b_user_id_like',
  'TE_switch_b_user_id_reply',
  'TE_switch_b_user_id_retweet',
  'TE_switch_b_user_id_retweet_comment',
  'TE_valid_tweet_id_like',
  'TE_valid_tweet_id_reply',
  'TE_valid_tweet_id_retweet',
  'TE_valid_tweet_id_retweet_comment',
  'CE_valid_b_user_id'],
 'retweet': ['TE_a_user_id_like',
  'TE_a_user_id_reply',
  'TE_a_user_id_retweet',
  'TE_a_user_id_retweet_comment',
  'TE_b_is_verified_tweet_type_like',
  'TE_b_is_verified_tweet_type_reply',
  'TE_b_is_verified_tweet_type_retweet',
  'TE_b_is_verified_tweet_type_retweet_comment',
  'TE_b_user_id_a_user_id_like',
  'TE_b_user_id_a_user_id_reply',
  'TE_b_user_id_a_user_id_retweet',
  'TE_b_user_id_a_user_id_retweet_comment',
  'TE_b_user_id_like',
  'TE_b_user_id_reply',
  'TE_b_user_id_retweet',
  'TE_b_user_id_retweet_comment',
  'TE_b_user_id_tweet_type_language_like',
  'TE_b_user_id_tweet_type_language_reply',
  'TE_b_user_id_tweet_type_language_retweet',
  'TE_b_user_id_tweet_type_language_retweet_comment',
  'TE_tw_original_user0_tweet_type_language_like',
  'TE_tw_original_user0_tweet_type_language_reply',
  'TE_tw_original_user0_tweet_type_language_retweet',
  'TE_tw_original_user0_tweet_type_language_retweet_comment',
  'TE_tw_original_user1_tweet_type_language_like',
  'TE_tw_original_user1_tweet_type_language_reply',
  'TE_tw_original_user1_tweet_type_language_retweet',
  'TE_tw_original_user1_tweet_type_language_retweet_comment',
  'TE_tweet_type_like',
  'TE_tweet_type_reply',
  'TE_tweet_type_retweet',
  'TE_tweet_type_retweet_comment',
  'a_ff_rate',
  'a_follower_count',
  'a_following_count',
  'a_is_verified',
  'ab_age_dff',
  'ab_age_rate',
  'ab_fer_rate',
  'ab_fing_rate',
  'b_ff_rate',
  'b_follower_count',
  'b_following_count',
  'b_follows_a',
  'b_is_verified',
  'dt_dow',
  'dt_hour',
  'dt_minute',
  'language',
  'len_domains',
  'len_hashtags',
  'len_links',
  'media',
  'tw_count_at',
  'tw_count_capital_words',
  'tw_count_char',
  'tw_count_excl_quest_marks',
  'tw_count_special1',
  'tw_count_words',
  'tw_last_quest',
  'tw_len_gif',
  'tw_len_media',
  'tw_len_photo',
  'tw_len_quest',
  'tw_len_retweet',
  'tw_len_rt',
  'tw_len_token',
  'tw_len_video',
  'tweet_type',
  'CNN_reply',
  'CNN_retweet',
  'CNN_retweet_comment',
  'CNN_like',
  'CXGB_reply',
  'CXGB_retweet',
  'CXGB_retweet_comment',
  'CXGB_like',
  'BONN_reply',
  'BONN_retweet',
  'BONN_retweet_comment',
  'BONN_like',
  'XGB_reply',
  'XGB_retweet',
  'XGB_retweet_comment',
  'XGB_like',
  'BNN_reply',
  'BNN_retweet',
  'BNN_retweet_comment',
  'BNN_like',
  'GXGB_reply',
  'GXGB_retweet',
  'GXGB_retweet_comment',
  'GXGB_like',
  'TE_valid_a_user_id_like',
  'TE_valid_a_user_id_reply',
  'TE_valid_a_user_id_retweet',
  'TE_valid_a_user_id_retweet_comment',
  'TE_valid_b_user_id_like',
  'TE_valid_b_user_id_reply',
  'TE_valid_b_user_id_retweet',
  'TE_valid_b_user_id_retweet_comment',
  'TE_switch_a_user_id_like',
  'TE_switch_a_user_id_reply',
  'TE_switch_a_user_id_retweet',
  'TE_switch_a_user_id_retweet_comment',
  'TE_switch_b_user_id_like',
  'TE_switch_b_user_id_reply',
  'TE_switch_b_user_id_retweet',
  'TE_switch_b_user_id_retweet_comment',
  'TE_valid_tweet_id_like',
  'TE_valid_tweet_id_reply',
  'TE_valid_tweet_id_retweet',
  'TE_valid_tweet_id_retweet_comment',
  'CE_valid_b_user_id'],
 'retweet_comment': ['TE_a_user_id_like',
  'TE_a_user_id_reply',
  'TE_a_user_id_retweet',
  'TE_a_user_id_retweet_comment',
  'TE_b_is_verified_tweet_type_like',
  'TE_b_is_verified_tweet_type_reply',
  'TE_b_is_verified_tweet_type_retweet',
  'TE_b_is_verified_tweet_type_retweet_comment',
  'TE_b_user_id_a_user_id_like',
  'TE_b_user_id_a_user_id_reply',
  'TE_b_user_id_a_user_id_retweet',
  'TE_b_user_id_a_user_id_retweet_comment',
  'TE_b_user_id_like',
  'TE_b_user_id_reply',
  'TE_b_user_id_retweet',
  'TE_b_user_id_retweet_comment',
  'TE_b_user_id_tweet_type_language_like',
  'TE_b_user_id_tweet_type_language_reply',
  'TE_b_user_id_tweet_type_language_retweet',
  'TE_b_user_id_tweet_type_language_retweet_comment',
  'TE_tw_original_user0_tweet_type_language_like',
  'TE_tw_original_user0_tweet_type_language_reply',
  'TE_tw_original_user0_tweet_type_language_retweet',
  'TE_tw_original_user0_tweet_type_language_retweet_comment',
  'TE_tw_original_user1_tweet_type_language_like',
  'TE_tw_original_user1_tweet_type_language_reply',
  'TE_tw_original_user1_tweet_type_language_retweet',
  'TE_tw_original_user1_tweet_type_language_retweet_comment',
  'TE_tweet_type_like',
  'TE_tweet_type_reply',
  'TE_tweet_type_retweet',
  'TE_tweet_type_retweet_comment',
  'a_ff_rate',
  'a_follower_count',
  'a_following_count',
  'a_is_verified',
  'ab_age_dff',
  'ab_age_rate',
  'ab_fer_rate',
  'ab_fing_rate',
  'b_ff_rate',
  'b_follower_count',
  'b_following_count',
  'b_follows_a',
  'b_is_verified',
  'dt_dow',
  'dt_hour',
  'dt_minute',
  'language',
  'len_domains',
  'len_hashtags',
  'len_links',
  'media',
  'tw_count_at',
  'tw_count_capital_words',
  'tw_count_char',
  'tw_count_excl_quest_marks',
  'tw_count_special1',
  'tw_count_words',
  'tw_last_quest',
  'tw_len_gif',
  'tw_len_media',
  'tw_len_photo',
  'tw_len_quest',
  'tw_len_retweet',
  'tw_len_rt',
  'tw_len_token',
  'tw_len_video',
  'tweet_type',
  'CNN_reply',
  'CNN_retweet',
  'CNN_retweet_comment',
  'CNN_like',
  'CXGB_reply',
  'CXGB_retweet',
  'CXGB_retweet_comment',
  'CXGB_like',
  'BONN_reply',
  'BONN_retweet',
  'BONN_retweet_comment',
  'BONN_like',
  'XGB_reply',
  'XGB_retweet',
  'XGB_retweet_comment',
  'XGB_like',
  'BNN_reply',
  'BNN_retweet',
  'BNN_retweet_comment',
  'BNN_like',
  'GXGB_reply',
  'GXGB_retweet',
  'GXGB_retweet_comment',
  'GXGB_like',
  'TE_valid_a_user_id_like',
  'TE_valid_a_user_id_reply',
  'TE_valid_a_user_id_retweet',
  'TE_valid_a_user_id_retweet_comment',
  'TE_valid_b_user_id_like',
  'TE_valid_b_user_id_reply',
  'TE_valid_b_user_id_retweet',
  'TE_valid_b_user_id_retweet_comment',
  'TE_switch_a_user_id_like',
  'TE_switch_a_user_id_reply',
  'TE_switch_a_user_id_retweet',
  'TE_switch_a_user_id_retweet_comment',
  'TE_switch_b_user_id_like',
  'TE_switch_b_user_id_reply',
  'TE_switch_b_user_id_retweet',
  'TE_switch_b_user_id_retweet_comment',
  'TE_valid_tweet_id_like',
  'TE_valid_tweet_id_reply',
  'TE_valid_tweet_id_retweet',
  'TE_valid_tweet_id_retweet_comment',
  'CE_valid_b_user_id'],
 'like': ['TE_a_user_id_like',
  'TE_a_user_id_reply',
  'TE_a_user_id_retweet',
  'TE_a_user_id_retweet_comment',
  'TE_b_is_verified_tweet_type_like',
  'TE_b_is_verified_tweet_type_reply',
  'TE_b_is_verified_tweet_type_retweet',
  'TE_b_is_verified_tweet_type_retweet_comment',
  'TE_b_user_id_a_user_id_like',
  'TE_b_user_id_a_user_id_reply',
  'TE_b_user_id_a_user_id_retweet',
  'TE_b_user_id_a_user_id_retweet_comment',
  'TE_b_user_id_like',
  'TE_b_user_id_reply',
  'TE_b_user_id_retweet',
  'TE_b_user_id_retweet_comment',
  'TE_b_user_id_tweet_type_language_like',
  'TE_b_user_id_tweet_type_language_reply',
  'TE_b_user_id_tweet_type_language_retweet',
  'TE_b_user_id_tweet_type_language_retweet_comment',
  'TE_tw_original_user0_tweet_type_language_like',
  'TE_tw_original_user0_tweet_type_language_reply',
  'TE_tw_original_user0_tweet_type_language_retweet',
  'TE_tw_original_user0_tweet_type_language_retweet_comment',
  'TE_tw_original_user1_tweet_type_language_like',
  'TE_tw_original_user1_tweet_type_language_reply',
  'TE_tw_original_user1_tweet_type_language_retweet',
  'TE_tw_original_user1_tweet_type_language_retweet_comment',
  'TE_tweet_type_like',
  'TE_tweet_type_reply',
  'TE_tweet_type_retweet',
  'TE_tweet_type_retweet_comment',
  'a_ff_rate',
  'a_follower_count',
  'a_following_count',
  'a_is_verified',
  'ab_age_dff',
  'ab_age_rate',
  'ab_fer_rate',
  'ab_fing_rate',
  'b_ff_rate',
  'b_follower_count',
  'b_following_count',
  'b_follows_a',
  'b_is_verified',
  'dt_dow',
  'dt_hour',
  'dt_minute',
  'language',
  'len_domains',
  'len_hashtags',
  'len_links',
  'media',
  'tw_count_at',
  'tw_count_capital_words',
  'tw_count_char',
  'tw_count_excl_quest_marks',
  'tw_count_special1',
  'tw_count_words',
  'tw_last_quest',
  'tw_len_gif',
  'tw_len_media',
  'tw_len_photo',
  'tw_len_quest',
  'tw_len_retweet',
  'tw_len_rt',
  'tw_len_token',
  'tw_len_video',
  'tweet_type',
  'CNN_reply',
  'CNN_retweet',
  'CNN_retweet_comment',
  'CNN_like',
  'CXGB_reply',
  'CXGB_retweet',
  'CXGB_retweet_comment',
  'CXGB_like',
  'BONN_reply',
  'BONN_retweet',
  'BONN_retweet_comment',
  'BONN_like',
  'XGB_reply',
  'XGB_retweet',
  'XGB_retweet_comment',
  'XGB_like',
  'BNN_reply',
  'BNN_retweet',
  'BNN_retweet_comment',
  'BNN_like',
  'GXGB_reply',
  'GXGB_retweet',
  'GXGB_retweet_comment',
  'GXGB_like',
  'TE_valid_a_user_id_like',
  'TE_valid_a_user_id_reply',
  'TE_valid_a_user_id_retweet',
  'TE_valid_a_user_id_retweet_comment',
  'TE_valid_b_user_id_like',
  'TE_valid_b_user_id_reply',
  'TE_valid_b_user_id_retweet',
  'TE_valid_b_user_id_retweet_comment',
  'TE_switch_a_user_id_like',
  'TE_switch_a_user_id_reply',
  'TE_switch_a_user_id_retweet',
  'TE_switch_a_user_id_retweet_comment',
  'TE_switch_b_user_id_like',
  'TE_switch_b_user_id_reply',
  'TE_switch_b_user_id_retweet',
  'TE_switch_b_user_id_retweet_comment',
  'TE_valid_tweet_id_like',
  'TE_valid_tweet_id_reply',
  'TE_valid_tweet_id_retweet',
  'TE_valid_tweet_id_retweet_comment',
  'CE_valid_b_user_id']}

pathresults = './'

pred_files = [
    'results_benny_xgb.parquet',
    'results_benny_nn.parquet',
    'results_bo_nn.parquet',
    'results-chris-nn.pq',
    'results-chris-xgb.pq',
    'results-giba.parquet'
]

colnames = [
    ['tweet_id_org','b_user_id_org','XGB_reply','XGB_retweet','XGB_retweet_comment','XGB_like'],
    ['tweet_id_org','b_user_id_org','BNN_reply','BNN_retweet','BNN_retweet_comment','BNN_like'],
    ['tweet_id_org','b_user_id_org','BONN_reply','BONN_retweet','BONN_retweet_comment','BONN_like'],
    ['tweet_id_org','b_user_id_org','CNN_reply','CNN_retweet','CNN_retweet_comment','CNN_like'],
    ['tweet_id_org','b_user_id_org','CXGB_reply','CXGB_retweet','CXGB_retweet_comment','CXGB_like'],
    ['tweet_id_org','b_user_id_org','GXGB_reply','GXGB_retweet','GXGB_retweet_comment','GXGB_like']
]

NO_BAGS = 3
SUBMISSION = os.path.exists('./test/')
if SUBMISSION:
    path = './test/'
else:
    path = './test/'

labels = ['reply', 'retweet', 'retweet_comment', 'like']

def index_merge(df, df_tmp, col):
    key = '_'.join(col)
    if key in ['a_user_id', 'b_user_id']:
        return(
            df.merge(df_tmp[df_tmp.index.isin(df[col[0]])], 
                     how='left', 
                     left_on=col, 
                     right_index=True
                    )
        )
    elif key in [
        'b_user_id_a_user_id',
        'b_user_id_tweet_type_language',
        'domains_language_b_follows_a_tweet_type_media_a_is_verified',
        'tw_original_user0_tweet_type_language',
        'tw_original_user1_tweet_type_language'
    ]:
        return(
            df.merge(df_tmp[df_tmp.index.get_level_values(0).isin(df[col[0]])], 
                     how='left', 
                     left_on=col, 
                     right_index=True
                    )
        )
    elif key in ['b_is_verified_tweet_type', 
                 'media_tweet_type_language', 
                 'media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a',
                 'tweet_type']:
        return(
            df.merge(df_tmp, 
                     how='left', 
                     left_on=col, 
                     right_index=True
                    )
        )

def add_stage1(fn, TE_files_valid, CE_files_valid):
    psmooth=20
    means = {}
    means['reply'] = 0.02846728456689906
    means['like'] = 0.3968895210408169
    means['retweet'] = 0.08769760903336701
    means['retweet_comment'] = 0.006918407917391091
    means_valid = pickle.load(open('means_valid.pickle', 'rb'))
    df = cudf.read_parquet(fn)
    print(df.shape)
    for i, pred_file in enumerate(pred_files):
        print(pred_file)
        if pred_file[-3:] == 'csv':
            sub0 = cudf.read_csv(pathresults + pred_file, 
                   header=None, dtype={0:object,1:object,
                                       2:np.float32,3:np.float32,
                                       4:np.float32,5:np.float32})
        else:
            sub0 = cudf.read_parquet(pathresults + pred_file)
            
        sub0.columns = colnames[i]
        df = df.merge(sub0, how='left', on=['tweet_id_org', 'b_user_id_org'])
        del sub0
        gc.collect()
    # Add missing switch column
    filename = './TE_submission_opt_index/b_user_id.parquet'
    df_tmp = cudf.read_parquet(filename)
    col = list(df_tmp.index.names)
    col_rest = list(df_tmp.columns)
    dfcolsnew = []
    for co in col:
        if co == 'b_user_id':
            dfcolsnew.append('a_user_id')
        elif co == 'a_user_id':
            dfcolsnew.append('b_user_id')
        else:
            dfcolsnew.append(co)
    df_tmp = df_tmp.reset_index()
    df_tmp.columns = dfcolsnew + col_rest
    df = df.merge(df_tmp, on=dfcolsnew, how='left')
    col = dfcolsnew
    #col_rest = list(df_tmp.columns)
    for key in means.keys():
        df['TE_switch_' + '_'.join(col) + '_' + key] = (((df[key + '_sum'])+means[key]*psmooth)/(df['reply_count']+psmooth))
        df['TE_switch_' + '_'.join(col) + '_' + key] = df['TE_switch_' + '_'.join(col) + '_' + key].fillna(np.float32(means[key])).round(3)
    df.drop(col_rest, inplace=True, axis=1)
    del df_tmp
    gc.collect()
    for i, file in enumerate(TE_files_valid):
        df_tmp = cudf.read_parquet(file)
        col = [x for x in df_tmp.columns if not('reply' in x or ('retweet' in x and 'tw_len_retweet' not in x) or '_retweet_comment' in x or 'like' in x)]
        col_rest = [x for x in df_tmp.columns if x not in col]
        df = df.merge(df_tmp, on=col, how='left')
        for key in means.keys():
            df['TE_valid_' + '_'.join(col) + '_' + key] = (
                ((df[key + '_sum'])+means_valid[key]*psmooth)/(df['reply_count']+psmooth)
            )
            if col[0]=='a_user_id' and key=='like':
                df.loc[df['reply_count']<=1000, 'TE_valid_' + '_'.join(col) + '_' + key] = None
            df['TE_valid_' + '_'.join(col) + '_' + key] = df['TE_valid_' + '_'.join(col) + '_' + key].fillna(np.float32(means_valid[key])).round(3)
        df.drop(col_rest, inplace=True, axis=1)
        col = [x for x in df_tmp.columns if not('reply' in x or ('retweet' in x and 'tw_len_retweet' not in x) or '_retweet_comment' in x or 'like' in x)]
        del df_tmp
        gc.collect()
    # NN valid:
    print('CE valid')
    for i, file in enumerate(CE_files_valid):
        df_tmp = cudf.read_parquet(file)
        col = [list(df_tmp.columns)[0]]
        df_tmp.columns = col + ['CE_valid_' + col[0]]
        df = df.merge(df_tmp, on=col, how='left')
        df['CE_valid_' + col[0]] = df['CE_valid_' + col[0]].fillna(1)
        del df_tmp
        gc.collect()
    print(df.shape)
    df.to_parquet(fn.replace('/test_convert_TE/', '/test_convert_TE_stack/'))
    del df
    gc.collect()

CE_cols = [
    ['b_user_id']
]

def collect_CE(df):
    for col in CE_cols:
        print(col)
        df_tmp = df[
            col + ['reply']
             ].groupby(col).count()
        df_tmp = df_tmp.reset_index()
        df_tmp.columns = col + ['CE_' + col[0]]
        df_tmp = df_tmp.to_parquet('./CE_valid/' + '_'.join(col) + '.parquet')
    del df_tmp
    gc.collect()

if __name__ == "__main__":
    print("Stage 2")
    print(path)
    os.system('mkdir ' + path.replace('/test/', '/test_convert_TE_stack/'))
    os.system('mkdir ' + './CE_valid')
    start_time = time.time()
    print('Collect CE')
    files = glob.glob(path.replace('/test/', '/test_convert_TE/') + 'part*')
    dftmp = cudf.read_parquet(files[0], columns=[item for sublist in CE_cols for item in sublist] + ['reply'])
    collect_CE(dftmp)
    del dftmp
    gc.collect()
    CE_files_valid = glob.glob('./CE_valid/*.parquet')
    TE_files_valid = glob.glob('./TE_valid/*.parquet')
    print('Adding Stage1')
    for file in files:
        add_stage1(file, TE_files_valid, CE_files_valid)
    gc.collect()
    time_fe = time.time()-start_time
    print("Prediction")
    files = glob.glob(path.replace('/test/', '/test_convert_TE_stack/') + 'part*')
    print(files)
    loadout = []
    for label in ['like', 'reply', 'retweet', 'retweet_comment']:
        loadout = loadout + train_features_stage2[label]
    loadout = sorted(list(set(loadout)))+['group', 'tweet_id_org', 'b_user_id_org']
    start_time = time.time()
    for file in files:
        print(file)
        df = cudf.read_parquet(file, columns=loadout)
        for label in labels:
            print('Label:' + str(label))
            df[label] = 0
            xgb_features = train_features_stage2[label]
            dvalid = cupy.array(df[xgb_features].values.astype('float32'), order='C' )
            for bag in range(3):
                model = pickle.load(open('./models_stacked/model_' + label + '_' + str(bag) + '.pickle', 'rb'))
                model['booster'].save_model('xgboostmodel' + label + '.pickle')
                filmodel = ForestInference.load('xgboostmodel' + label + '.pickle', output_class=True)
                pred = filmodel.predict_proba(dvalid)[:,1]            
                df[label] = (df[label].values+(pred/3))
            del pred, dvalid
            gc.collect()
        df[['tweet_id_org', 
            'b_user_id_org', 
            'reply', 
            'retweet', 
            'retweet_comment', 
            'like']].to_parquet('results_benny_stage2.parquet', header=False, index=False)
        del df
        gc.collect()
    time_xgb = time.time()-start_time
    print('Time stage2 fe: ' + str(time_fe))
    print('Time stage2 xgb:  ' + str(time_xgb))
    if False:
        print('results_stage2')
        df = pd.read_csv('results_benny_stage2.csv', header=None)
        print(df.isna().sum())
        print(df.head())
        print(df.shape)
        print(df[[2,3,4,5]].mean())