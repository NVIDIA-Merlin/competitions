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

# RESTRICT TO GPU 0 ONLY
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[3]:


# RESTRICT TENSORFLOW TO 1GB OF GPU RAM
# SO THAT WE HAVE 15GB RAM FOR RAPIDS
LIMIT = 20
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*LIMIT)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
print('We will restrict TensorFlow to max %iGB GPU RAM'%LIMIT)
print('then RAPIDS can use %iGB GPU RAM'%(32-LIMIT))


import os
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import gc
import glob
import re, time
import hashlib
import pickle

import tensorflow as tf
print('TF',tf.__version__)
import cudf, cupy
print('cudf',cudf.__version__)

#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

ISSUB = True # set to False to test in the local machine

if ISSUB:
    BASEDIR = './test_proc3/'
    BASEDIR2 = './test_tokens/'
else:
    BASEDIR = '/raid/RecSys/recsys2021/val0bbp/'
    BASEDIR2 = '/raid/RecSys/recsys2021/val0bbt/'


# BUILD MODEL
    
EMB_SIZE = 96
EMB_SIZE2 = 96 * 2
TOK_SIZE = 48

class OwnEmb(tf.keras.layers.Layer):
    def __init__(self, num_input, num_output, name=None, **kwargs):
        self.num_input = num_input
        self.num_output = num_output
        super(OwnEmb, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.embedding_tables = self.add_weight(
            trainable=True,
            initializer="glorot_normal",
            shape=(self.num_input, self.num_output)
        )
        self.built = True

    def call(self, x):
        return tf.gather(self.embedding_tables, x)

    def compute_output_shape(self, input_shapes):
        return (input_shape[0], self.num_output)

def build_model(sz1,sz2,sz3,sz4):
    inp = tf.keras.layers.Input(shape=(sz1,))
    inp2 = tf.keras.layers.Input(shape=(TOK_SIZE,))

    embeds = []
    embeds.append( OwnEmb(sz4+1,EMB_SIZE) ) # USER_ID
    
    embeds.append( OwnEmb(3,2) ) # TWEET_TYPE
    embeds.append( OwnEmb(66,10) ) # LANGUAGE
    embeds.append( OwnEmb(14,4) ) # MEDIA
    embeds.append( OwnEmb(7,4) ) # DT_DOW
       
    # USERS
    a_user = embeds[0](tf.cast(inp[:, 0], tf.int64))
    b_user = embeds[0](tf.cast(inp[:, 1], tf.int64))
    
    # USER INTERACTION
    a_embed = tf.keras.layers.Concatenate()([a_user,inp[:,-sz2:-sz2+2],inp[:,-1:]])
    a_embed = tf.keras.layers.Dense(EMB_SIZE,activation='tanh')(a_embed)
    a_dot_b = tf.keras.layers.Dot(axes=-1,normalize=True)([a_embed,b_user])
        
    # CAT FEATURE EMBEDDINGS
    embeds2 = []    
    for k in range(2,sz3):
        embeds2.append( embeds[k-1](tf.cast(inp[:, k], tf.int64)) )
    x1 = tf.keras.layers.Concatenate()(embeds2)
        
    # TWEET TOKEN EMBEDDINGS
    embeds3 = []
    word_emb = OwnEmb(119548,EMB_SIZE2)
    for k in range(TOK_SIZE):
        embeds3.append( word_emb(tf.cast(inp2[:, k], tf.int64)) )
    x2 = tf.keras.layers.Average()(embeds3)
    
    # USER INTERACT WITH TWEET
    tweet_embed = tf.keras.layers.Concatenate()([x1,x2,inp[:,-sz2:-3]])
    tweet_embed = tf.keras.layers.Dense(EMB_SIZE,activation='tanh')(tweet_embed)
    b_dot_tweet = tf.keras.layers.Dot(axes=-1,normalize=True)([tweet_embed,b_user])

    # NUMERICAL FEATURES
    x = tf.keras.layers.Concatenate()(
        [a_user,b_user,a_dot_b,b_dot_tweet,x1,x2,inp[:,-sz2:]])
    
    HIDDEN_SIZE = 256+64
    LAYERS = 3
    
    for k in range(LAYERS):
        x = tf.keras.layers.Dense(HIDDEN_SIZE)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
    # CONCAT WITH MATRIX FACTORIZATION
    x = tf.keras.layers.Concatenate()([a_dot_b,b_dot_tweet,x])
    
    x = tf.keras.layers.Dense(4,activation='sigmoid',dtype='float32')(x)
    model = tf.keras.models.Model(inputs=[inp,inp2],outputs=x)
    
    opt = tf.keras.optimizers.Adam(lr=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer = opt)
    return model


def standardize(df,IDX,IDX2):
    
    # LOG STANDARDIZE
    LOG_FEATS = ['a_follower_count','a_following_count','b_follower_count','b_following_count']
    for f in LOG_FEATS:
        name = 'LOG_'+f
        print(name,', ',end='')
        df[name] = cupy.log1p( df[f].values, dtype='float32' )
        del df[f]
        gc.collect()
        
    # NORM STANDARDIZE
    NORM_FEATS = ['LOG_a_follower_count','LOG_a_following_count',
                  'dt_minute','len_hashtags','len_links','len_domains']

    NORM_FEATS += ['tw_len_media', 'tw_len_photo','tw_len_video', 'tw_len_gif', 
                  'tw_len_quest', 'tw_len_token',
                  'tw_count_capital_words', 'tw_count_excl_quest_marks',
                  'tw_count_special1', 'tw_count_hash', 'tw_last_quest', 
                  'tw_len_retweet', 'tw_len_rt', 'tw_count_at', 'tw_count_words', 'tw_count_char',
                  'tw_rt_count_words', 'tw_rt_count_char']

    NORM_FEATS += ['LOG_b_follower_count','LOG_b_following_count','b_follows_a']
    
    data = cudf.read_csv('standardize_%i.csv'%VER).set_index('feature')
    for f in NORM_FEATS:
        name = 'NORM_'+f
        print(name,', ',end='')
        mn = data.loc[f,'mean']
        st = data.loc[f,'std']
        df[name] = ((df[f].values - mn) /st).astype('float32')
        del df[f]
        gc.collect()
        
    # USER LABEL ENCODE
    print(); print('user1 ct',len(IDX))
    user_map = cudf.DataFrame()
    user_map['a_user_id3'] = IDX
    user_map['a_user_id'] = cupy.arange(len(IDX))+1

    print('user2 ct',len(IDX2))
    user_map2 = cudf.DataFrame()
    user_map2['a_user_id3'] = IDX2
    user_map2['a_user_id2'] = cupy.arange(len(IDX2))+1
    
    # PREPARE FOR MERGE
    df = df.rename({'a_user_id':'a_user_id3','b_user_id':'b_user_id3'},axis=1)
    df['idx'] = cupy.arange(len(df))
    
    df = df.merge(user_map,on='a_user_id3',how='left')
    df['a_user_id'] = df['a_user_id'].fillna(0).astype('int32')
    
    user_map.columns = ['b_user_id3','b_user_id']
    df = df.merge(user_map,on='b_user_id3',how='left')
    df['b_user_id'] = df['b_user_id'].fillna(0).astype('int32')
    
    df = df.merge(user_map2,on='a_user_id3',how='left')
    df['a_user_id2'] = df['a_user_id2'].fillna(0).astype('int32')
    del df['a_user_id3']
    
    user_map2.columns = ['b_user_id3','b_user_id2']
    df = df.merge(user_map2,on='b_user_id3',how='left')
    df['b_user_id2'] = df['b_user_id2'].fillna(0).astype('int32')
    del df['b_user_id3']
    
    df = df.sort_values('idx')
    del df['idx']
    
    return df

from datetime import datetime

def evaluate_test_set():
    print('-----------------------------------------------------------')
    print('evaluate_test_set')
        
    part_files = sorted( glob.glob(BASEDIR+'*.parquet') )
    print(part_files)
    print()
    
    targets = ['reply','retweet','retweet_comment','like']
    FEATURES = ['a_user_id', 'b_user_id', 'tweet_type', 'language', 'media',
       'dt_dow', 'NORM_LOG_a_follower_count',
       'NORM_LOG_a_following_count', 'NORM_dt_minute',
       'NORM_len_hashtags', 'NORM_len_links', 'NORM_len_domains',
       'NORM_tw_len_media', 'NORM_tw_len_photo', 'NORM_tw_len_video',
       'NORM_tw_len_gif', 'NORM_tw_len_quest', 'NORM_tw_len_token',
       'NORM_tw_count_capital_words', 'NORM_tw_count_excl_quest_marks',
       'NORM_tw_count_special1', 'NORM_tw_count_hash',
       'NORM_tw_last_quest', 'NORM_tw_len_retweet', 'NORM_tw_len_rt',
       'NORM_tw_count_at', 'NORM_tw_count_words', 'NORM_tw_count_char',
       'NORM_tw_rt_count_words', 'NORM_tw_rt_count_char',
       'NORM_LOG_b_follower_count', 'NORM_LOG_b_following_count',
       'NORM_b_follows_a']
    FEATURES2 = ['a_user_id2', 'b_user_id2'] + FEATURES[2:]
    CAT_FEATS = [f for f in FEATURES if not 'NORM' in f]
    NORM_FEATS = [f for f in FEATURES if 'NORM' in f]
    USER_IDS = np.load('user_map_%i.npy'%VER)
    USER_IDS2 = np.load('user_map_%i.npy'%VER2)
    
    model = build_model(len(FEATURES),len(NORM_FEATS),len(CAT_FEATS),len(USER_IDS))
    model.load_weights('./nn%i.h5'%VER)
    
    model2 = build_model(len(FEATURES2),len(NORM_FEATS),len(CAT_FEATS),len(USER_IDS2))
    model2.load_weights('./nn%i.h5'%VER2)
    
    COLS = ['a_user_id32','b_user_id32']
    COLS += [f.replace('NORM_','').replace('LOG_','') for f in FEATURES][2:]
    COLS += ['tweet_id_org','b_user_id_org']

    checkpointA = time.time()
            
    # READ IN ALL ORIGINAL PROCESSED DATA
    all_parts = []
    for file in part_files:
        df = cudf.read_parquet(file,columns=COLS)
        df.columns = ['a_user_id','b_user_id'] + COLS[2:]
        print(file, df.shape)
        all_parts.append(df)
    df = cudf.concat(all_parts,axis=0,ignore_index=True)
            
    # LOG TRANSFORM, NORMALIZE, AND LABEL ENCODE
    print('Standardize features...')
    df = standardize(df,USER_IDS,USER_IDS2)
            
    # READ IN ALL ORIGINAL TOKENS
    print('Loading tokens...')
    all_parts = []
    for file in part_files:
        t_name = BASEDIR2 + file.split("/")[-1].split(".")[0] +".npy"
        test_tokens = cupy.load(t_name)[:,:TOK_SIZE]
        all_parts.append(test_tokens)
    test_tokens = cupy.concatenate(all_parts,axis=0)
            
    checkpointB = time.time()
    print('### Preprocess Elapsed',(checkpointB-checkpointA),'seconds')
            
    # CONVERT CUDF AND CUPY TO DLPACK TO TF TENSOR
    X1 = cupy.array( df[FEATURES].values, order='C' ).toDlpack()
    X1 = tf.experimental.dlpack.from_dlpack( X1 )
    X3 = tf.experimental.dlpack.from_dlpack( test_tokens.toDlpack() )
    # PREDICT WITH TENSORFLOW
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    preds = model.predict([X1, X3], verbose=1, batch_size=1024*128)/2.0
    
    del X1
    gc.collect()

    X2 = cupy.array( df[FEATURES2].values, order='C').toDlpack()
    X2 = tf.experimental.dlpack.from_dlpack( X2 )
    preds += model2.predict([X2, X3], verbose=1, batch_size=1024*128)/2.0
    #preds = model.predict([df[FEATURES].to_pandas(), cupy.asnumpy(test_tokens)], verbose=1, batch_size=1024*32)/2.0
    #preds += model2.predict([df[FEATURES2].to_pandas(), cupy.asnumpy(test_tokens)], verbose=1, batch_size=1024*32)/2.0                             
    for k in range(4):
        df[targets[k]] = preds[:,k]
    df[['tweet_id_org', 
        'b_user_id_org', 
        'reply', 
        'retweet', 
        'retweet_comment', 
        'like']].to_parquet('results-chris-nn-new.pq', header=False, index=False)
            
            
    checkpointC = time.time()
    print('### NN Infer Elapsed',(checkpointC-checkpointB),'seconds')
            
    print('### NN Total Elapsed',(checkpointC-checkpointA),'seconds')
    print()
            
    del df
    gc.collect()


if __name__ == "__main__":
############################################################################
############################################################################
############################################################################
    VER = 86; VER2 = 87
    
    start = time.time()
    x = evaluate_test_set() # Infer and write to results-chris-nn.csv
    end = time.time()
    print('#'*25)
    print('NN Total Script Elapsed',end-start,'seconds')

