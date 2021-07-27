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

#!/env/bin/python

### https://recsys-twitter.com/leaderboard/latest
### https://hub.docker.com/repository/docker/titericz/recsys2021
### titericz/recsys_2021_build070521:v1

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import xgboost as xgb
import cudf as pd # Use CUDF instead of Pandas for speed of the light merging
import numpy as np
import joblib
import gc
import glob
import re
import hashlib
import pickle
import time
import cupy
from cuml import ForestInference

starttime = time.time()

os.makedirs('tmp_giba', exist_ok=True)
BASEDIR = './tmp_giba/'

COLS = ['tweet_id', 'media', 'links', 'domains', 'tweet_type',
       'language', 'timestamp', 'a_user_id', 'a_follower_count',
       'a_following_count', 'a_is_verified', 'a_account_creation', 'b_user_id',
       'b_follower_count', 'b_following_count', 'b_is_verified',
       'b_account_creation', 'b_follows_a', 'like', 'reply', 'retweet',
       'retweet_comment', 'tweet_id_org', 'b_user_id_org', 'tw_len_token',
       'text', 'tw_len_media', 'tw_len_photo', 'tw_len_video', 'tw_len_gif',
       'tw_len_quest', 'tw_count_capital_words', 'tw_count_excl_quest_marks',
       'tw_count_special1', 'tw_count_hash', 'tw_last_quest', 'tw_len_retweet',
       'tw_len_rt', 'tw_count_at', 'tw_count_words', 'tw_count_char',
       'tw_rt_count_words', 'tw_rt_count_char', 'tw_original_user0',
       'tw_original_user1', 'tw_original_user2', 'tw_rt_user0', 'tw_word0',
       'tw_word1', 'tw_tweet', 'group', 'dt_day', 'dt_dow', 'dt_minute',
       'len_hashtags', 'len_links', 'len_domains', 'a_user_id32',
       'b_user_id32', 'decline']

def merge_features_encoding(part_files):

    all_parts = []
    for file in part_files:
        df = pd.read_parquet(file,columns=COLS)
        print(file, df.shape)
        all_parts.append(df)
    df = pd.concat(all_parts,axis=0,ignore_index=True)
    print(df.shape)        
    del all_parts; _ = gc.collect()
    
    df['decline'] = df['decline']//7.19
    df['a_follower_count'] = df['a_follower_count'] // 819
    df['a_following_count'] = df['a_following_count'] // 712
    df['b_follower_count'] = df['b_follower_count'] // 819
    df['b_following_count'] = df['b_following_count'] // 712

    dt = pd.read_parquet('GIBA_TEMAPS/a_user_id-tweet_type.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/b_user_id-tweet_type.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/b_user_id.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/tw_rt_user0.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/tw_original_user0-tweet_type.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/tw_original_user1-tweet_type.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/tw_word0-tweet_type.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt

    dt = pd.read_parquet('GIBA_TEMAPS/a_follower_count-a_following_count-b_follower_count-b_following_count-tweet_type-language-b_follows_a.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt
    
    dt = pd.read_parquet('GIBA_TEMAPS/media-language-tweet_type-a_is_verified-b_is_verified-b_follows_a-tw_last_quest-decline-group.parquet')
    features = list(dt.index.names)
    df = df.merge(dt, left_on=features, right_index=True, how='left')
    del dt
    
    for f in [
        'sums_te_reply-a_user_id-tweet_type',
        'counts_te_reply-a_user_id-tweet_type',
        'sums_te_retweet-a_user_id-tweet_type',
        'sums_te_retweet_comment-a_user_id-tweet_type',
        'sums_te_like-a_user_id-tweet_type',
        'sums_te_reply-b_user_id-tweet_type',
        'counts_te_reply-b_user_id-tweet_type',
        'sums_te_retweet-b_user_id-tweet_type',
        'sums_te_retweet_comment-b_user_id-tweet_type',
        'sums_te_like-b_user_id-tweet_type', 'multi_reply', 'multi_retweet',
        'multi_retweet_comment', 'multi_like', 'multi_counts', 'ouser0_reply',
        'ouser0_retweet', 'ouser0_retweet_comment', 'ouser0_like',
        'ouser0_counts', 'ouser1_reply', 'ouser1_retweet',
        'ouser1_retweet_comment', 'ouser1_like', 'ouser1_counts', 'word_reply',
        'word_retweet', 'word_retweet_comment', 'word_like', 'word_counts',
        'rtuser0_reply', 'rtuser0_retweet', 'rtuser0_retweet_comment',
        'rtuser0_like', 'rtuser0_counts', 'follow_reply', 'follow_retweet',
        'follow_retweet_comment', 'follow_like', 'follow_counts',
        'sums_b_user_reply', 'sums_b_user_retweet',
        'sums_b_user_retweet_comment', 'sums_b_user_like', 'counts_b_user_id'      
        ]:
        df[f] = df[f].astype(np.float32)
    _ = gc.collect()
        
    return df


def load_XGB(fn):
    return ForestInference.load(fn, output_class=True)

def evaluate_test_set(df):
    print('-----------------------------------------------------------')
    print('Evaluate_test_set')
    
    REPLY_FEATURES = pickle.load(open('GIBA_TEMAPS/xgb/xgbmodels-features-reply.pickle', 'rb'))
    RETWEET_FEATURES = pickle.load(open('GIBA_TEMAPS/xgb/xgbmodels-features-retweet.pickle', 'rb'))
    QUOTE_FEATURES = pickle.load(open('GIBA_TEMAPS/xgb/xgbmodels-features-retweet_comment.pickle', 'rb'))
    LIKE_FEATURES = pickle.load(open('GIBA_TEMAPS/xgb/xgbmodels-features-like.pickle', 'rb'))

    print('Loading XGB weights:')
    MODELS = {}
    MODELS['reply'] = []
    MODELS['retweet'] = []
    MODELS['retweet_comment'] = []
    MODELS['like'] = []
    for i in range(5):
        MODELS['reply'].append(load_XGB( 'GIBA_TEMAPS/xgb/model_reply_'+str(i)+'.pickle'  ))
        MODELS['retweet'].append(load_XGB( 'GIBA_TEMAPS/xgb/model_retweet_'+str(i)+'.pickle'))
        MODELS['retweet_comment'].append(load_XGB( 'GIBA_TEMAPS/xgb/model_retweet_comment_'+str(i)+'.pickle'))
        MODELS['like'].append(load_XGB( 'GIBA_TEMAPS/xgb/model_like_'+str(i)+'.pickle'))
    print('Loading XGB models done!!!')

    for feat in np.unique(REPLY_FEATURES+RETWEET_FEATURES+QUOTE_FEATURES+LIKE_FEATURES):
        if df[feat].dtype != 'float32':
            df[feat] = df[feat].astype('float32')
        df[feat] = df[feat].fillna(-999.)
    print(df.shape)

    df['reply'] = 0.
    df['retweet'] = 0.
    df['retweet_comment'] = 0.
    df['like'] = 0.
    for group in range(5):
        
        dvalid0 = cupy.array( df.loc[df.group==group,REPLY_FEATURES].values.astype('float32'), order='C' )
        dvalid1 = cupy.array( df.loc[df.group==group,RETWEET_FEATURES].values.astype('float32'), order='C' )
        dvalid2 = cupy.array( df.loc[df.group==group,QUOTE_FEATURES].values.astype('float32'), order='C' )
        dvalid3 = cupy.array( df.loc[df.group==group,LIKE_FEATURES].values.astype('float32'), order='C' )

        df.loc[df.group==group,'reply'] = MODELS['reply'][group].predict_proba(dvalid0)[:,1]
        df.loc[df.group==group,'retweet'] = MODELS['retweet'][group].predict_proba(dvalid1)[:,1]
        df.loc[df.group==group,'retweet_comment'] = MODELS['retweet_comment'][group].predict_proba(dvalid2)[:,1]
        df.loc[df.group==group,'like'] = MODELS['like'][group].predict_proba(dvalid3)[:,1]
    
    df[['tweet_id_org', 
        'b_user_id_org', 
        'reply', 
        'retweet', 
        'retweet_comment', 
        'like']].to_parquet("results-giba.parquet", index=True) 


if __name__ == "__main__":
############################################################################
############################################################################
############################################################################
    print('-----------------------------------------------------------')
    part_files = glob.glob('./test_proc3/*.parquet')
    print(part_files)
    df = merge_features_encoding(part_files)
    print('elapsed:', time.time()-starttime)
    print()
    
    evaluate_test_set(df) # write to results.csv 
    print('elapsed:', time.time()-starttime)
    print()
    
    #print( pd.read_csv('results-giba.csv').iloc[:,2:].mean() )
    