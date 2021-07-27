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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import glob

import torch
import torch.nn as nn

from utils_GPU_v2 import *

class CustomDataset(Dataset):
    """Simple dataset class for dataloader"""
    def __init__(self, df, tokens, max_len, OTHERS_NUM1):
        """Initialize the CustomDataset"""
        self.tokens = tokens[:, :max_len]
        self.b_user_cat = df[B_USER_CAT].values
        self.b_user_num = df[B_USER_NUM].values
        self.tweet_cat = df[TWEET_CAT].values
        self.tweet_num = df[TWEET_NUM].values
        self.other_cat = df[OTHERS_CAT].values
        self.other_num = df[OTHERS_NUM1].values
        self.musers = df[['tw_original_user0_', 'tw_original_user1_', 'tw_original_user2_']].values
    
    def __len__(self):
        """Return the total length of the dataset"""
        dataset_size = self.tokens.shape[0]
        return dataset_size
  
    def __getitem__(self, idx):
        """Return the batch given the indices"""
        return (self.tokens[idx].astype(np.int64), 
                self.b_user_cat[idx].astype(np.int64),
                self.tweet_cat[idx].astype(np.int64),
                self.other_cat[idx].astype(np.int64),
                self.b_user_num[idx].astype(np.float32),
                self.tweet_num[idx].astype(np.float32),
                self.other_num[idx].astype(np.float32),
                self.musers[idx].astype(np.int64)
               )

def index_merge_simple(df, df_tmp, col):
#     return(
#         df.merge(df_tmp[df_tmp.index.isin(df[col])], 
#                  how='left', 
#                  left_on=col, 
#                  right_index=True
#                 )
#     )
    return(
        df.merge(df_tmp, on=col, how='left')
    )
    
def preprocess_nn(fn, dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu, 
                  NUM_STATS, NUM_LOG_COLS, NUM_COLS, NUM_TE):
    df = cudf.read_parquet(fn)
    df['date'] = cudf.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['date'].dt.hour
    df['a_ff_rate'] = (df['a_following_count'] / (1+df['a_follower_count'])).astype('float32')
    df['b_ff_rate'] = (df['b_follower_count']  / (1+df['b_following_count'])).astype('float32')
    df['ab_fing_rate'] = (df['a_following_count'] / (1+df['b_following_count'])).astype('float32')
    df['ab_fer_rate'] = (df['a_follower_count'] / (1+df['b_follower_count'])).astype('float32')
    df['tw_count_at'].clip(lower=cupy.int16(0), upper=cupy.int16(6), inplace=True)
    df['tw_count_special1'].clip(lower=cupy.int16(0), upper=cupy.int16(6), inplace=True)
    df['tw_len_quest'].clip(lower=cupy.int8(0), upper=cupy.int8(6), inplace=True)
    df['tw_len_retweet'].clip(lower=cupy.int8(0), upper=cupy.int8(1), inplace=True)
    df['tw_len_rt'].clip(lower=cupy.int8(0), upper=cupy.int8(1), inplace=True)
    col = 'b_user_id'
    dfuseremb.columns = [col, col + '_']
    df = index_merge_simple(df, dfuseremb, col)
    df[col + '_'] = df[col + '_'].fillna(0)
    col = 'hashtags'
    dfhashtags.columns = [col, col + '_']
    df = index_merge_simple(df, dfhashtags, col)
    df[col + '_'] = df[col + '_'].fillna(0)
    col = 'domains'
    dfdomains.columns = [col, col + '_']
    df = index_merge_simple(df, dfdomains, col)
    df[col + '_'] = df[col + '_'].fillna(0)
    col = 'tw_rt_user0'
    dfrtu.columns = [col, col + '_']
    df = index_merge_simple(df, dfrtu, col)
    df[col + '_'] = df[col + '_'].fillna(0)
    for col in ['tw_original_user0', 'tw_original_user1', 'tw_original_user2']:
        dfmuseremb.columns = [col, col + '_']
        df = index_merge_simple(df, dfmuseremb, col)
        df[col + '_'] = df[col + '_'].fillna(0)
    for col in NUM_LOG_COLS:
        df[col] = (cupy.log((df[col]+1).astype('float32'))-NUM_STATS[col][0])/NUM_STATS[col][1]
    for col in NUM_COLS:
        df[col] = ((df[col]).astype('float32')-NUM_STATS[col][0])/NUM_STATS[col][1]
    for col in NUM_TE:
        df[col] = (df[col]-NUM_STATS[col][1])/(NUM_STATS[col][0]-NUM_STATS[col][1])
    df.to_parquet(fn.replace('/test_convert_TE/', '/test_convert_NN/'))

import cupy    
from torch.utils.dlpack import from_dlpack, to_dlpack

def process_epoch(
    dataloader,
    model,
    train=False,
    optimizer=None,
    total_loss=0.0,
    n=0,
    y_list=[],
    y_pred_list=[],
    loss_weights=[1.0,1.0,1.0,1.0],
    batch_size=1024,
    rest=[]
):
    model.train(mode=train)
    #with open("results_benny_nn.csv", "a+") as outfile:
    for idx, batch in enumerate(iter(dataloader)):
        with torch.no_grad():
            #n+=batch[0].shape[0]
            tokens = batch[0][:,0:42]
            b_user_cat = batch[0][:,42:(42+len(B_USER_CAT))]
            tweet_cat = batch[0][:,(42+len(B_USER_CAT)):(42+len(B_USER_CAT)+len(TWEET_CAT))]
            other_cat = batch[0][:,(42+len(B_USER_CAT)+len(TWEET_CAT)):(42+len(B_USER_CAT)+len(TWEET_CAT)+len(OTHERS_CAT))]
            m_users_cat = batch[0][:,(42+len(B_USER_CAT)+len(TWEET_CAT)+len(OTHERS_CAT)):(42+len(B_USER_CAT)+len(TWEET_CAT)+len(OTHERS_CAT)+3)]
            b_user_num = batch[1][:,0:(len(B_USER_NUM))]
            tweet_num = batch[1][:,(len(B_USER_NUM)):(len(B_USER_NUM)+len(TWEET_NUM))]
            other_num = batch[1][:,(len(B_USER_NUM)+len(TWEET_NUM)):(len(B_USER_NUM)+len(TWEET_NUM)+len(OTHERS_NUM + NUM_TE))]
            y_pred = model(tokens, tweet_cat, tweet_num, None, None, b_user_cat, b_user_num, other_cat, other_num, m_users_cat)[0]
            #y_list.append(y.detach().cpu().numpy())
            #y_pred_list.append(torch.sigmoid(y_pred).detach().cpu().numpy())
            y_pred_list.append(torch.sigmoid(y_pred).detach())
            #np_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
            #np_ids = rest[batch_size*idx:batch_size*(idx+1)]
            #np.savetxt(outfile, np.hstack([np_ids, np_pred]), delimiter=",", fmt="%s,%s,%10.6f,%10.6f,%10.6f,%10.6f")
            #total_loss += loss.detach().cpu().item()*n
    y_pred_out = torch.cat(y_pred_list)
    y_pred_out = cupy.fromDlpack(to_dlpack(y_pred_out))
    rest['reply'] = y_pred_out[:, 0]
    rest['retweet'] = y_pred_out[:, 1]
    rest['retweet_comment'] = y_pred_out[:, 2]
    rest['like'] = y_pred_out[:, 3]
    rest[['tweet_id_org', 
          'b_user_id_org', 
          'reply', 
          'retweet', 
          'retweet_comment', 
          'like']].to_parquet('results_benny_nn.parquet', header=False, index=False)
    return total_loss, n, y_list, []