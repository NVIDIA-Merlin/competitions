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
import glob
import pickle
import gc
import pandas as pd
#from transformers import BertTokenizer
from utils_GPU_v2 import extract_feature, add_TE, train_features, get_colnames

from nn_utils_GPU_v2 import preprocess_nn, CustomDataset, process_epoch

import xgboost as xgb
from tqdm.notebook import tqdm

import torch

from torch.utils.data import DataLoader

from utils_GPU_v2 import *

import time
import cudf
import cupy

import nvtabular as nvt
from nvtabular.loader.torch import TorchAsyncItr

from cuml import ForestInference

NO_BAGS = 3
SUBMISSION = os.path.exists('./test/')
if SUBMISSION:
    path = './test/'
else:
    path = './test/'

labels = ['reply', 'retweet', 'retweet_comment', 'like']

if __name__ == "__main__":
    print("Load tokenizer")
    print(path)
    os.system('mkdir ' + path.replace('/test/', '/test_convert/'))
    os.system('mkdir ' + path.replace('/test/', '/test_split/'))
    os.system('mkdir ' + path.replace('/test/', '/test_convert_TE/'))
    os.system('mkdir ' + path.replace('/test/', '/test_convert_NN/'))
    start_time = time.time()
    time_splitting = time.time()-start_time
    start_time = time.time()
    time_extract_feature = time.time()-start_time
    print("Target Encoding")
    start_time = time.time()
    te_mapping = []
    TE_files = sorted(glob.glob('./TE_submission_opt_index/*.parquet'))
    TE_files = [
        './TE_submission_opt_index/b_user_id_tweet_type_language.parquet',
        './TE_submission_opt_index/b_user_id_a_user_id.parquet',
        './TE_submission_opt_index/a_user_id.parquet',
        './TE_submission_opt_index/b_is_verified_tweet_type.parquet',
        './TE_submission_opt_index/b_user_id.parquet',
        './TE_submission_opt_index/domains_language_b_follows_a_tweet_type_media_a_is_verified.parquet',
        './TE_submission_opt_index/media_tweet_type_language.parquet',
        './TE_submission_opt_index/media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a.parquet',
        './TE_submission_opt_index/tw_original_user0_tweet_type_language.parquet',
        './TE_submission_opt_index/tw_original_user1_tweet_type_language.parquet',
        './TE_submission_opt_index/tweet_type.parquet'
    ]
#     for file in TE_files:
#         df_tmp = cudf.read_parquet(file)
#         te_mapping.append(df_tmp)
#         gc.collect()
    files = glob.glob(path.replace('/test/', '/test_proc3/') + 'part*')
    print("Target Encoding per file")
    for file in [0]:
        print(file)
        add_TE(files, TE_files)
    del TE_files
    gc.collect()
    time_te = time.time()-start_time
    print("Prediction")
    start_time = time.time()
    files = glob.glob(path.replace('/test/', '/test_convert_TE/') + 'part*')
    print(files)
    loadout = []
    for label in ['like', 'reply', 'retweet', 'retweet_comment']:
        loadout = loadout + train_features[label]
    loadout = sorted(list(set(loadout)))+['group', 'tweet_id_org', 'b_user_id_org']
    for file in files:
        print(file)
        #df = cudf.concat([cudf.read_parquet(file, columns=loadout) for file in files],axis=0,ignore_index=True)
        df = cudf.read_parquet(file, columns=loadout)
        df['quantile'] = 0
        for label in labels:
            print('Label:' + str(label))
            df[label] = 0
            if label in ['like', 'retweet']:
                model = pickle.load(open('./models_TE_weird/model_' + str(label) + '.pickle', 'rb'))
            if label in ['reply', 'retweet_comment']:
                model = pickle.load(open('./models_TE_leaveoneout_bothweeks/model_' + str(label) + '.pickle', 'rb'))
            model['booster'].save_model('xgboostmodel' + label + '.pickle')
            filmodel = ForestInference.load('xgboostmodel' + label + '.pickle', output_class=True)
            xgb_features = train_features[label]
            dvalid = cupy.array(df[xgb_features].values.astype('float32'), order='C' )
            pred = filmodel.predict_proba(dvalid)[:,1]            
#             bst = model['booster']
#             xgb_features = train_features[label]
#             dftmp = df[xgb_features].copy()
#             dftmp.columns = ['f' + str(x) for x in range(len(dftmp.columns))]
#             #dtest = xgb.DMatrix(data=dftmp)
#             pred = bst.inplace_predict(dftmp)
            df[label] = pred
            del pred, dvalid
            gc.collect()
        df[['tweet_id_org', 
            'b_user_id_org', 
            'reply', 
            'retweet', 
            'retweet_comment', 
            'like']].to_parquet('results_benny_xgb.parquet', header=False, index=False)
#         with open("results_benny_xgb.csv", "a+") as outfile:
#             df[['tweet_id_org', 
#                 'b_user_id_org', 
#                 'reply', 
#                 'retweet', 
#                 'retweet_comment', 
#                 'like']].to_csv(outfile, header=False, index=False)
        del df
        gc.collect()
    time_pred_xgb = time.time()-start_time
    start_time = time.time()
    dfuseremb = cudf.read_parquet('./NN_encoding_submissions_index/abusercount.parquet')
    dfuseremb = dfuseremb.reset_index()
    dfmuseremb = cudf.read_parquet('./NN_encoding_submissions_index/muser_id.parquet')
    dfmuseremb = dfmuseremb.reset_index()
    dfhashtags = cudf.read_parquet('./NN_encoding_submissions_index/hashtags.parquet')
    dfhashtags = dfhashtags.reset_index()
    dfdomains = cudf.read_parquet('./NN_encoding_submissions_index/domains.parquet')
    dfdomains = dfdomains.reset_index()
    dfrtu = cudf.read_parquet('./NN_encoding_submissions_index/tw_rt_user0.parquet')
    dfrtu = dfrtu.reset_index()
    files = glob.glob(path.replace('/test/', '/test_convert_TE/') + 'part*')
    for file in files:
        print(file)
        preprocess_nn(file, dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu, NUM_STATS, NUM_LOG_COLS, NUM_COLS, NUM_TE)
    del dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu
    gc.collect()
    time_prep_nn = time.time()-start_time
    files = glob.glob(path.replace('/test/', '/test_convert_NN/') + 'part*')
    print("NN Prediction")
    start_time = time.time()
    model = torch.load('model_out.pth').cuda()
    outload = ['tw_original_user0_', 'tw_original_user1_', 'tw_original_user2_', 'tweet_id_org', 'b_user_id_org'] + B_USER_CAT + B_USER_NUM + TWEET_CAT + TWEET_NUM + OTHERS_CAT + OTHERS_NUM + NUM_TE
    bs_size = 1024*20
    for file in files:
        print(file)
        df = cudf.read_parquet(file, columns=outload+['idx'])
        df = df.sort_values('idx')
        token_files = sorted(glob.glob('./test_tokens/*.npy'))
        all_parts = []
        for t_file in token_files:
            print(t_file)
            test_tokens = cupy.load(t_file)[:,:42]
            all_parts.append(test_tokens)
        cptokens = cupy.concatenate(all_parts,axis=0)
        #cptokens = cupy.load(os.path.join('./test_tokens/' , file.split('/')[-1].replace('.parquet','.npy')))
        for i in range(42):
            df['text_tokens_' + str(i)] = cptokens[:, i]
        del cptokens, all_parts
        for col in ['text_tokens_' + str(i) for i in range(42)] + B_USER_CAT + TWEET_CAT + OTHERS_CAT + ['tw_original_user0_', 'tw_original_user1_', 'tw_original_user2_']:
            df[col] = df[col].astype(np.int64)
        for col in B_USER_NUM + TWEET_NUM + OTHERS_NUM + NUM_TE:
            df[col] = df[col].astype(np.float32)
        dict_rename = {}
        counter = 0
        for col in ['text_tokens_' + str(i) for i in range(42)] + B_USER_CAT + TWEET_CAT + OTHERS_CAT + ['tw_original_user0_', 'tw_original_user1_', 'tw_original_user2_']:
            dict_rename[col] = 'col1' + str(counter).zfill(4)
            counter+=1
        counter = 0
        for col in B_USER_NUM + TWEET_NUM + OTHERS_NUM + NUM_TE:
            dict_rename[col] = 'col2' + str(counter).zfill(4)
            counter+=1
        df.rename(columns=dict_rename, inplace=True)
        df.to_parquet('tmptest.parquet')
        train_loader = TorchAsyncItr(
            nvt.Dataset('tmptest.parquet', part_size="2048MB"),
            batch_size=bs_size,
            cats=[x for x in df.columns if 'col1' in x],
            conts=[x for x in df.columns if 'col2' in x],
            labels=[],
            shuffle=False
        )
        dftmp = df[['tweet_id_org', 'b_user_id_org']].copy()
        del df
        gc.collect()
        out = process_epoch(train_loader, model, False, y_list=[], 
                            y_pred_list=[], batch_size=bs_size, rest=dftmp)
        del train_loader
        gc.collect()
    time_pred_nn = time.time()-start_time
    print('Time splitting: ' + str(time_splitting))
    print('Time ext feat:  ' + str(time_extract_feature))
    print('Time target en: ' + str(time_te))
    print('Time XGB pred:  ' + str(time_pred_xgb))
    print('Time prep NN:   ' + str(time_prep_nn))
    print('Time pred NN:   ' + str(time_pred_nn))
    #os.system('rm -r test_convert_NN')
    if False:
        print('results_benny_xgb')
        df = pd.read_csv('results_benny_xgb.csv', header=None)
        print(df.isna().sum())
        print(df.head())
        print(df.shape)
        print(df[[2,3,4,5]].mean())
        print('results_benny_nn')
        df = pd.read_csv('results_benny_nn.csv', header=None)
        print(df.isna().sum())
        print(df.head())
        print(df.shape)
        print(df[[2,3,4,5]].mean())
