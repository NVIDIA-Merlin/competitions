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

#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/env/bin/python

# titericz/recsys_2021_build070521:v1

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[3]:


import os, time
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import gc
import glob
import re
import hashlib
import pickle

import cudf, cupy
from cuml import ForestInference
print('cudf version',cudf.__version__)

# In[2]:


ISSUB = True # set to False to test in the local machine

if ISSUB:
    BASEDIR = './test_proc3/'
    OUTDIR = './test_proc_chris/'
else:
    BASEDIR = '/raid/RecSys/recsys2021/val0aap/'
    OUTDIR = '/raid/RecSys/recsys2021/val0aap2/'

# In[3]:

COLS = ['a_user_id32','b_user_id32', 'tweet_id_org','b_user_id_org', 
       'tweet_type', 'language', 'media', 'tw_word0',
       'b_follows_a', 'a_follower_count', 'a_following_count',
       'b_follower_count', 'b_following_count', 'dt_dow', 'dt_minute',
       'len_hashtags', 'len_links', 'len_domains', 'tw_len_media',
       'tw_len_photo', 'tw_len_video', 'tw_len_gif', 'tw_len_quest',
       'tw_len_token', 'tw_count_capital_words',
       'tw_count_excl_quest_marks', 'tw_count_special1', 'tw_count_hash',
       'tw_last_quest', 'tw_len_retweet', 'tw_len_rt', 'tw_count_at',
       'tw_count_words', 'tw_count_char', 'tw_rt_count_words',
       'tw_rt_count_char']

def preprocess_merge():

    targets = ['reply','retweet','retweet_comment','like']
    
    # FULL TRAIN 1 BYTE MEANS
    M = {'reply': -120,
         'retweet': -105,
         'retweet_comment': -126,
         'like': -27}
            
    print('-----------------------------------------------------------')
    print('preprocess_te')
        
    # preprocessed test parts
    part_files = sorted( glob.glob(BASEDIR+'*.parquet') )
    print(part_files)
        
    feats = [['a_user_id'],['b_user_id'],['tweet_type'],['language'],['media'],
             ['tw_word0'],
             ['a_user_id','language','tweet_type'],
             ['b_user_id','language','tweet_type'],
             ['language','tweet_type','media'],
             ['a_user_id','b_user_id']]
        
    smooths = [20]*len(feats); filts = [1]*len(feats)
    fnames = ['_'.join(x)+f'_s{y}_f{z}' for x,y,z in zip(feats,smooths,filts)]
    
    # LOAD ALL ORIGINAL PREPROCESS FILES
    all_parts = []
    for file in part_files:
        df = cudf.read_parquet(file,columns=COLS)
        print(file,df.shape)
        df.columns = ['a_user_id','b_user_id'] + COLS[2:]
        all_parts.append(df)
    df = cudf.concat(all_parts,axis=0,ignore_index=True)
    print(df.shape)
        
    ###################
    # TE FEATURES MERGE
    for feat,fname in zip(feats,fnames):
        dt = cudf.read_parquet('./te146/' + fname + '.parquet')
        df = df.merge(dt, on=feat, how='left')
        for target in targets:    
            name = 'TE_'+'_'.join(feat)+'_'+target
            df[name] = df[name].fillna(M[target]).astype('int8')
        gc.collect()
        print(df.shape,', ',end='')
            
    ###################
    # FEATURE ENGINEER
    df['ff_a_ratio'] = (df['a_follower_count'] / (df['a_following_count']+1)).astype('float32')
    df['ff_b_ratio'] = (df['b_follower_count'] / (df['b_following_count']+1)).astype('float32')
                    
    print(df.shape, ', ',end='')
        
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_parquet(os.path.join(OUTDIR, file.split('/')[-1] ))
        
    gc.collect() 
    print(); print()
        
    return df


def evaluate_test_set(df):
    print('-----------------------------------------------------------')
    print('evaluate_test_set')
        
    print('Loading XGB models...')
    targets = ['reply','retweet','retweet_comment','like']
    FEATS = ['b_follows_a', 'a_follower_count', 'a_following_count',
       'b_follower_count', 'b_following_count', 'dt_dow', 'dt_minute',
       'len_hashtags', 'len_links', 'len_domains', 'tw_len_media',
       'tw_len_photo', 'tw_len_video', 'tw_len_gif', 'tw_len_quest',
       'tw_len_token', 'tw_count_capital_words',
       'tw_count_excl_quest_marks', 'tw_count_special1', 'tw_count_hash',
       'tw_last_quest', 'tw_len_retweet', 'tw_len_rt', 'tw_count_at',
       'tw_count_words', 'tw_count_char', 'tw_rt_count_words',
       'tw_rt_count_char', 'TE_a_user_id_reply', 'TE_a_user_id_retweet',
       'TE_a_user_id_retweet_comment', 'TE_a_user_id_like',
       'TE_b_user_id_reply', 'TE_b_user_id_retweet',
       'TE_b_user_id_retweet_comment', 'TE_b_user_id_like',
       'TE_tweet_type_reply', 'TE_tweet_type_retweet',
       'TE_tweet_type_retweet_comment', 'TE_tweet_type_like',
       'TE_language_reply', 'TE_language_retweet',
       'TE_language_retweet_comment', 'TE_language_like',
       'TE_media_reply', 'TE_media_retweet', 'TE_media_retweet_comment',
       'TE_media_like', 'TE_tw_word0_reply', 'TE_tw_word0_retweet',
       'TE_tw_word0_retweet_comment', 'TE_tw_word0_like',
       'TE_a_user_id_language_tweet_type_reply',
       'TE_a_user_id_language_tweet_type_retweet',
       'TE_a_user_id_language_tweet_type_retweet_comment',
       'TE_a_user_id_language_tweet_type_like',
       'TE_b_user_id_language_tweet_type_reply',
       'TE_b_user_id_language_tweet_type_retweet',
       'TE_b_user_id_language_tweet_type_retweet_comment',
       'TE_b_user_id_language_tweet_type_like',
       'TE_language_tweet_type_media_reply',
       'TE_language_tweet_type_media_retweet',
       'TE_language_tweet_type_media_retweet_comment',
       'TE_language_tweet_type_media_like',
       'TE_a_user_id_b_user_id_reply', 'TE_a_user_id_b_user_id_retweet',
       'TE_a_user_id_b_user_id_retweet_comment',
       'TE_a_user_id_b_user_id_like', 'ff_a_ratio', 'ff_b_ratio']
    
    models_a = []
    for k in range(4):
        model_path = './m146/XGB_%s_a'%(targets[k])
        model = ForestInference.load(model_path, output_class=True, 
                                    threads_per_tree=16,storage_type='dense',algo='BATCH_TREE_REORG',
                                    n_items=2,blocks_per_sm=5)
        models_a.append(model)
        
    models_b = []
    for k in range(4):
        model_path = './m146/XGB_%s_b'%(targets[k])
        model = ForestInference.load(model_path, output_class=True, 
                                    threads_per_tree=16,storage_type='dense',algo='BATCH_TREE_REORG',
                                    n_items=2,blocks_per_sm=5)
        models_b.append(model)
        
    models_c = []
    for k in range(4):
        model_path = './m146/XGB_%s_c'%(targets[k])
        model = ForestInference.load(model_path, output_class=True, 
                                    threads_per_tree=16,storage_type='dense',algo='BATCH_TREE_REORG',
                                    n_items=2,blocks_per_sm=5)
        models_c.append(model)

    print(df.shape)
            
    dvalid = cupy.array( df[FEATS].values.astype('float32'), order='C' )
    preds = cupy.zeros((len(df),4),dtype='float32')
            
    checkA = time.time()
    for k in range(4):
        preds[:,k] += models_a[k].predict_proba(dvalid)[:,1]/3.
    for k in range(4):
        preds[:,k] += models_b[k].predict_proba(dvalid)[:,1]/3.
    for k in range(4):
        preds[:,k] += models_c[k].predict_proba(dvalid)[:,1]/3.
    checkB = time.time()
    print('### Inference',checkB-checkA,'seconds')
                                   
    for k in range(4):
        df[targets[k]] = preds[:,k]
    df[['tweet_id_org', 
        'b_user_id_org', 
        'reply', 
        'retweet', 
        'retweet_comment', 
        'like']].to_parquet('results-chris-xgb.pq', header=False, index=False)
                            
    del df
    gc.collect()

    return

if __name__ == "__main__":
############################################################################
############################################################################
############################################################################
    VER = 146; VER2 = 146
    
    checkpointA = time.time()
    df = preprocess_merge() # merge Target encoding features. 
    checkpointB = time.time()
    print('### XGB Stage 1 Preprocess Elapsed',(checkpointB-checkpointA),'seconds')
    
    evaluate_test_set(df) # write to results-xgb.csv
    checkpointC = time.time()
    print('### XGB Stage 1 Infer Elapsed',(checkpointC-checkpointB),'seconds')
    
    print('### XGB Stage 1 Total Elapsed',(checkpointC-checkpointA),'seconds')

