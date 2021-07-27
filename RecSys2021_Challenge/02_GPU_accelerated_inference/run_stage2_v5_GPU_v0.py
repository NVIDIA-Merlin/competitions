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
#import pandas as pd
import numpy as np
import joblib
import gc
import glob
import re, time
import hashlib
import pickle

import cudf, cupy
from cuml import ForestInference
print('cudf version',cudf.__version__)

ISSUB = True # set to False to test in the local machine

if ISSUB:
    BASEDIR0 = './test_proc3/'
    BASEDIR = './test_proc_chris/'
    #OUTDIR = './test_proc_chris2/'
    #OUTDIR2 = './test_proc_chris3/'
else:
    BASEDIR = '/raid/RecSys/recsys2021/val0aap/'
    #OUTDIR = '/raid/RecSys/recsys2021/val0aap2/'


def make_ce_maps():
    
    print('-----------------------------------------------------------')
    print('make_ce_maps')
        
    # preprocessed test parts
    part_files = sorted( glob.glob(BASEDIR0+'*.parquet') )
    print(part_files)
    
    # SAME LIST AS BELOW
    CE_MAPS = [['a_user_id32'], ['b_user_id32'],['language'],
               ['media'],['tw_word0'],['tweet_id'],
               ['a_user_id32','language','tweet_type'],
               ['b_user_id32','language','tweet_type'],
               ['language','tweet_type','media'],
               ['a_user_id32','b_user_id32']]
    
    CE_DTYPE = {'a_user_id32':32,'b_user_id32':32,'language':8,'media':8,
                'tw_word0':32,'tweet_id':64,'tweet_type':8}
    
    the_maps = {}
    
    for cols in CE_MAPS:
        name = 'CE_'+"_".join(cols)
        print(name,'  ',end='')
        parts = []
        for file in part_files:
            #print(file,', ',end='')
            df = cudf.read_parquet(file,columns=cols) 
            parts.append(df)
        df = cudf.concat(parts)
        tmp = df.groupby(cols)[cols[0]].agg('count').to_frame()
        tmp.columns = [name]
        tmp = tmp.reset_index()
        tmp[name] = tmp[name].astype('int32')
        for c in cols:
            tmp[c] = tmp[c].astype('int%i'%CE_DTYPE[c])
        print(tmp.shape)
        
        #os.makedirs(OUTDIR2, exist_ok=True)
        name = name.replace('32','')
        tmp.columns = [x.replace('32','') for x in tmp.columns.values]
        #tmp.to_parquet(os.path.join(OUTDIR2, name+'.parquet'))
        
        the_maps[name] = tmp.copy()

    return the_maps


def preprocess_merge(the_maps):

    targets = ['reply','retweet','retweet_comment','like']
    
    # REC SYS VAL 1 BYTE MEANS
    M = {'reply': -122,
         'retweet': -107,
         'retweet_comment': -126,
         'like': -25}
    
    # FULL TRAIN 1 BYTE MEANS
    M2 = {'reply': -120,
         'retweet': -105,
         'retweet_comment': -126,
         'like': -27}
            
    print('-----------------------------------------------------------')
    print('preprocess_te')
        
    # preprocessed test parts
    part_files = sorted( glob.glob(BASEDIR+'*.parquet') )
    print(part_files)
        
    feats = [['a_user_id'],['b_user_id'],['tweet_type'],['language'],['media'],
             ['tw_word0'],['tweet_id'],
             ['a_user_id','language','tweet_type'],
             ['b_user_id','language','tweet_type'],
             ['language','tweet_type','media'],
             ['a_user_id','b_user_id']]
    
    stage1 = ['results-chris-xgb.pq','results-chris-nn.pq',
              'results_benny_xgb.parquet','results_benny_nn.parquet',
              'results_bo_nn.parquet']
    
    snames = ['chris_xgb','chris_nn','benny_xgb','benny_nn','bo_nn']
    
    # SAME LIST AS ABOVE
    CE_MAPS = [['a_user_id'], ['b_user_id'],['language'],
               ['media'],['tw_word0'],['tweet_id'],
               ['a_user_id','language','tweet_type'],
               ['b_user_id','language','tweet_type'],
               ['language','tweet_type','media'],
               ['a_user_id','b_user_id']]
        
    smooths = [20]*len(feats); filts = [1]*len(feats)
    fnames = ['_'.join(x)+f'_s{y}_f{z}' for x,y,z in zip(feats,smooths,filts)]
    
    for file in part_files:
        print(file)
        df = cudf.read_parquet(file)        
        print(df.shape)
        df['tweet_id'] = df['tweet_id_org'].str[-16:].str.hex_to_int().astype('int64')
         

        ###################
        # TE FEATURES MERGE
        for feat,fname in zip(feats,fnames):
            dt = cudf.read_parquet(f'te{VER}/' + fname + '.parquet')
            ##### UPDATE COLUMN NAMES
            cc = list(dt.columns.values)
            dt.columns = cc[:-4] + [f+'_val' for f in cc[-4:]]
            #####
            df = df.merge(dt, on=feat, how='left')
            for target in targets:    
                name = 'TE_'+'_'.join(feat)+'_'+target+'_val'
                df[name] = df[name].fillna(M[target]).astype('int8') #uses val mean
            gc.collect()
            print(df.shape)
            
        print('-----------------------------------------------------------')
        print('preprocess_reverse_te_from_train')
        
        dt = cudf.read_parquet('./te146/a_user_id_s20_f1.parquet')
        dt.columns = ['b_user_id', 'TE_a_user_id_reply3', 'TE_a_user_id_retweet3',
               'TE_a_user_id_retweet_comment3', 'TE_a_user_id_like3']
        df = df.merge(dt, on='b_user_id', how='left')
        for target in targets:    
            name = 'TE_a_user_id_'+target+'3'
            df[name] = df[name].fillna(M2[target]).astype('int8') #uses train mean
        gc.collect()
        print(df.shape)
        
        dt = cudf.read_parquet('./te146/b_user_id_s20_f1.parquet') 
        dt.columns = ['a_user_id', 'TE_a_user_id_reply2', 'TE_a_user_id_retweet2',
               'TE_a_user_id_retweet_comment2', 'TE_a_user_id_like2']
        df = df.merge(dt, on='a_user_id', how='left')
        for target in targets:    
            name = 'TE_a_user_id_'+target+'2'
            df[name] = df[name].fillna(M2[target]).astype('int8') #uses train mean
        gc.collect()
        print(df.shape)
                        
        print('-----------------------------------------------------------')
        print('preprocess_stage1_models')
        ###################
        # STAGE 1 MODEL MERGE
        for j,s1 in enumerate(stage1):
            time_a = time.time()
            print('merging',s1,'...')
            if s1[-3:]=='csv':
                dt = cudf.read_csv(s1,header=None,dtype={0:object,1:object,
                    2:np.float32,3:np.float32,4:np.float32,5:np.float32})
            else:
                dt = cudf.read_parquet(s1)
            targets = ['reply','retweet','retweet_comment','like']
            dt.columns = ['tweet_id_org','b_user_id_org'] + [snames[j]+'_%i'%k for k in range(4)]
            df = df.merge(dt, on=['tweet_id_org','b_user_id_org'], how='left')
            time_b = time.time()
            print('elapsed %.1f'%((time_b-time_a)),'seconds')
            gc.collect()
            print(df.shape)
            
        print('-----------------------------------------------------------')
        print('preprocess_ce and reverse_ce')
        ###################
        # CE MAPS MERGE   
        for cols in CE_MAPS:
            name = 'CE_'+"_".join(cols)
            print(name,'  ',end='')
            dt = the_maps[name]
            #dt = cudf.read_parquet(os.path.join(OUTDIR2, name+'.parquet'))
            df = df.merge(dt, on=cols, how='left')
            print(df.shape)
            
            if (len(cols)==1) & (cols[0]=='b_user_id'):
                dt.columns = ['a_user_id','CE2_b_user_id']
                df = df.merge(dt, on='a_user_id', how='left')
                df['CE2_b_user_id'] = df['CE2_b_user_id'].fillna(1e6).astype('int32')
                print(df.shape)
            if (len(cols)==1) & (cols[0]=='a_user_id'):
                dt.columns = ['b_user_id','CE2_a_user_id']
                df = df.merge(dt, on='b_user_id', how='left')
                df['CE2_a_user_id'] = df['CE2_a_user_id'].fillna(1e6).astype('int32')
                print(df.shape)
                  
                
        print() 
        print(df.shape); print(df.head())
        
        #os.makedirs(OUTDIR, exist_ok=True)
        del df['tweet_id']
        #df.to_parquet(os.path.join(OUTDIR, file.split('/')[-1] ))
        
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
       'TE_a_user_id_b_user_id_like', 'ff_a_ratio', 'ff_b_ratio',
       'chris_xgb_0', 'chris_nn_0', 'chris_xgb_1', 'chris_nn_1',
       'chris_xgb_2', 'chris_nn_2', 'chris_xgb_3', 'chris_nn_3',
       'bo_nn_0', 'bo_nn_1', 'bo_nn_2', 'bo_nn_3', 'benny_nn_0',
       'benny_nn_1', 'benny_nn_2', 'benny_nn_3', 'benny_xgb_0',
       'benny_xgb_1', 'benny_xgb_2', 'benny_xgb_3',
       'TE_a_user_id_reply_val', 'TE_a_user_id_retweet_val',
       'TE_a_user_id_retweet_comment_val', 'TE_a_user_id_like_val',
       'TE_b_user_id_reply_val', 'TE_b_user_id_retweet_val',
       'TE_b_user_id_retweet_comment_val', 'TE_b_user_id_like_val',
       'TE_tweet_type_reply_val', 'TE_tweet_type_retweet_val',
       'TE_tweet_type_retweet_comment_val', 'TE_tweet_type_like_val',
       'TE_language_reply_val', 'TE_language_retweet_val',
       'TE_language_retweet_comment_val', 'TE_language_like_val',
       'TE_media_reply_val', 'TE_media_retweet_val',
       'TE_media_retweet_comment_val', 'TE_media_like_val',
       'TE_tw_word0_reply_val', 'TE_tw_word0_retweet_val',
       'TE_tw_word0_retweet_comment_val', 'TE_tw_word0_like_val',
       'TE_tweet_id_reply_val', 'TE_tweet_id_retweet_val',
       'TE_tweet_id_retweet_comment_val', 'TE_tweet_id_like_val',
       'TE_a_user_id_language_tweet_type_reply_val',
       'TE_a_user_id_language_tweet_type_retweet_val',
       'TE_a_user_id_language_tweet_type_retweet_comment_val',
       'TE_a_user_id_language_tweet_type_like_val',
       'TE_b_user_id_language_tweet_type_reply_val',
       'TE_b_user_id_language_tweet_type_retweet_val',
       'TE_b_user_id_language_tweet_type_retweet_comment_val',
       'TE_b_user_id_language_tweet_type_like_val',
       'TE_language_tweet_type_media_reply_val',
       'TE_language_tweet_type_media_retweet_val',
       'TE_language_tweet_type_media_retweet_comment_val',
       'TE_language_tweet_type_media_like_val',
       'TE_a_user_id_b_user_id_reply_val',
       'TE_a_user_id_b_user_id_retweet_val',
       'TE_a_user_id_b_user_id_retweet_comment_val',
       'TE_a_user_id_b_user_id_like_val', 'TE_a_user_id_reply2',
       'TE_a_user_id_retweet2', 'TE_a_user_id_retweet_comment2',
       'TE_a_user_id_like2', 'TE_a_user_id_reply3',
       'TE_a_user_id_retweet3', 'TE_a_user_id_retweet_comment3',
       'TE_a_user_id_like3', 'CE_a_user_id', 'CE2_a_user_id',
       'CE_b_user_id', 'CE2_b_user_id', 'CE_language', 'CE_media',
       'CE_tw_word0', 'CE_tweet_id', 'CE_a_user_id_language_tweet_type',
       'CE_b_user_id_language_tweet_type', 'CE_language_tweet_type_media',
       'CE_a_user_id_b_user_id']
        
    models_a = []
    for k in range(4):
        model_path = 'm%i/XGB_%s_a'%(VER,targets[k])
        model = ForestInference.load(model_path, output_class=True, 
                                    threads_per_tree=16,storage_type='dense',algo='BATCH_TREE_REORG',
                                    n_items=2,blocks_per_sm=4)
        models_a.append(model)
        
    models_b = []
    for k in range(4):
        model_path = 'm%i/XGB_%s_b'%(VER,targets[k])
        model = ForestInference.load(model_path, output_class=True, 
                                    threads_per_tree=16,storage_type='dense',algo='BATCH_TREE_REORG',
                                    n_items=2,blocks_per_sm=4)
        models_b.append(model)
        
    models_c = []
    for k in range(4):
        model_path = 'm%i/XGB_%s_c'%(VER,targets[k])
        model = ForestInference.load(model_path, output_class=True, 
                                    threads_per_tree=16,storage_type='dense',algo='BATCH_TREE_REORG',
                                    n_items=2,blocks_per_sm=4)
        models_c.append(model)

    print(df.shape) #
            
    dvalid = cupy.array( df[FEATS].values.astype('float32'), order='C' )
    preds = cupy.zeros((len(df),4),dtype='float32')
            
    for k in range(4):
        preds[:,k] = models_a[k].predict_proba(dvalid)[:,1]/3.
    for k in range(4):
        preds[:,k] += models_b[k].predict_proba(dvalid)[:,1]/3.
    for k in range(4):
        preds[:,k] += models_c[k].predict_proba(dvalid)[:,1]/3.

                                           
    for k in range(4):
        df[targets[k]] = preds[:,k]
    df[['tweet_id_org', 
        'b_user_id_org', 
        'reply', 
        'retweet', 
        'retweet_comment', 
        'like']].to_parquet('results-chris-stage2.pq', header=False, index=False)
                                
    del df
    gc.collect()

if __name__ == "__main__":
############################################################################
############################################################################
############################################################################
    VER = 227
    
    checkpointA = time.time()
    the_maps = make_ce_maps()
    df = preprocess_merge(the_maps) # merge Target encoding features and stage 1
    checkpointB = time.time()
    print('### XGB Stage 2 Preprocess Elapsed',(checkpointB-checkpointA),'seconds')
    
    evaluate_test_set(df)
    checkpointC = time.time()
    print('### XGB Stage 2 Infer Elapsed',(checkpointC-checkpointB),'seconds')
    
    print('### XGB Stage 2 Total Elapsed',(checkpointC-checkpointA),'seconds')

