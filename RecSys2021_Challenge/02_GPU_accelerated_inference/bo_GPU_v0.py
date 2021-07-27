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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cudf
import cupy
import time
import glob
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset,DataLoader

from bo_utils import NUMERIC_COLUMNS #, read_norm_merge
from common_preprocess import extract_feature, split
from bo_model import Net, AllDataset

import nvtabular as nvt
from nvtabular.loader.torch import TorchAsyncItr
from torch.utils.dlpack import from_dlpack, to_dlpack

max_len_txt=48
batch_size=8192

device = 'cuda'
    
label_names = sorted(['reply', 'retweet', 'retweet_comment', 'like'])
CAT_COLUMNS = ['a_user_id','b_user_id','language','media','tweet_type']

time_model = 0.0
time_fe = 0.0

def read_norm_merge(path, split='train'):

    ddf = cudf.read_parquet(path)
    ddf['idx'] = cupy.arange(len(ddf))
    ddf['date'] = cudf.to_datetime(ddf['timestamp'], unit='s')  

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
        mapping = cudf.read_parquet(f'./categories/unique.{mapping_col}.parquet').reset_index()
        mapping.columns = ['index',col]
        ddf = ddf.merge(mapping, how='left', on=col).drop(columns=[col]).rename(columns={'index':col})
        ddf[col] = ddf[col].fillna(0).astype('int')   
        
    ddf = ddf.sort_values('idx')

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

    
if __name__ == "__main__":

    start_time = time.time()
    print("start bo_gpu.py")
    print('\n', time.ctime(), "Start loading weights")
    stime = time.time()
    model = Net(len(NUMERIC_COLUMNS), layers=[1024,256,64], 
            embedding_table_shapes={'a_user_id_b_user_id': (19688213, 128), 'language': (67, 16), 'media': (15, 16), 'tweet_type': (4, 16)})                    

    model_name = 'load_thr3_pos_1e-2_1e-4_best_fp16'

    sd = torch.load(f'{model_name}.pth', map_location=torch.device('cpu'))
    model.load_state_dict(sd, strict=True)
    model.eval()
    model = model.to(device)
    time_model += time.time()-stime
    print('\n', time.ctime(), f"model {model_name} loaded")

    files = sorted(glob.glob('./test_proc3/part*'))
    print(files)
    

    result_lst = []

    for file in files:
        print('\n', time.ctime(), file)
        stime = time.time()
        df = read_norm_merge(file, 'test')
        tokens = cupy.load(file.replace('/test_proc3/','/test_tokens/').replace('.parquet','.npy'))

        for i in range(max_len_txt):
            df['text_tokens_' + str(i)] = tokens[:, i]
        del tokens

        dict_rename = {}

        token_columns = ['text_tokens_' + str(i) for i in range(max_len_txt)]

        for i,col in enumerate(token_columns + CAT_COLUMNS):
            df[col] = df[col].astype(np.int64)
            dict_rename[col] = 'col_cat' + str(i).zfill(4)

        for i,col in enumerate(NUMERIC_COLUMNS):
            df[col] = df[col].astype(np.float32)
            dict_rename[col] = 'col_cont' + str(i).zfill(4)

        df.rename(columns=dict_rename, inplace=True)    
        time_fe += time.time()-stime
        stime = time.time()
        test_loader = TorchAsyncItr(
            nvt.Dataset(df[list(dict_rename.values())]),
            batch_size=batch_size,
            cats=[x for x in df.columns if 'col_cat' in x],
            conts=[x for x in df.columns if 'col_cont' in x],
            labels=[],
            shuffle=False
        )

        LOGITS = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                text_tok = batch[0][:,:max_len_txt]
                x_cat = batch[0][:,max_len_txt:]
                x_cont = batch[1]

                logits = model(x_cat, x_cont, text_tok )
                LOGITS.append(logits)
        LOGITS = torch.cat(LOGITS).sigmoid()
        LOGITS = cupy.fromDlpack(to_dlpack(LOGITS)) 

        for i,label in enumerate(['like', 'reply', 'retweet', 'retweet_comment']):
            df[label] = LOGITS[:,i]

        result_lst.append(df[['tweet_id_org', 
            'b_user_id_org', 
            'reply', 
            'retweet', 
            'retweet_comment', 
            'like']].copy())
        time_model += time.time()-stime
        del df,test_loader
    
    stime = time.time()
    cudf.concat(result_lst).to_parquet('results_bo_nn.parquet')
    time_model += time.time()-stime
    print('Time prep NN:   ' + str(time_fe))
    print('Time pred NN:   ' + str(time_model))
    print('\n', time.ctime(), "End")
    end_time = time.time()
    
    print(f"bo_gpu.py Total Time = {end_time-start_time:.0f} seconds")


