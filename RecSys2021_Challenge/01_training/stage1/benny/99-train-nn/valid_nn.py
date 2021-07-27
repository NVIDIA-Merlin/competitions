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

from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

import torch
import pandas as pd
import numpy as np
import cudf
import glob
import gc
import random
import time
import os
import cupy

import pickle

from nn import AllNN
from nn_utils import CustomDataset, compute_rce, process_epoch, preprocess_nn, compute_rce_fast
from utils import *

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import argparse

from sklearn.metrics import average_precision_score

DEBUG = False

nfreq = 15
BATCH_SIZE = 1024*4
pretrained_gru = True
berttiny=False
shared_emb=False
matrix_fact=False
hidden_dim=64
hidden_lay=2
max_len = 42
EPOCHS = 1
optimizer_type='Adam'
learningrate=0.1
useremb = 32
loss_weights = [1.0, 1.0, 1.0, 1.0]
useTE = False
useAvg = True
useSkip = True
useAux = False
auxweight = 1.0

my_parser = argparse.ArgumentParser(description='NN')
my_parser.add_argument('nfreq',
                       type=str
                      )
my_parser.add_argument('pretrainedgru',
                       type=str
                      )
my_parser.add_argument('sharedemb',
                       type=str
                      )
my_parser.add_argument('matrixfact',
                       type=str
                      )
my_parser.add_argument('useremb',
                       type=str
                      )
my_parser.add_argument('altloss',
                       type=str
                      )
my_parser.add_argument('dropout',
                       type=str
                      )
my_parser.add_argument('grudim',
                       type=str
                      )
my_parser.add_argument('grubi',
                       type=str
                      )
my_parser.add_argument('grulayers',
                       type=str
                      )
my_parser.add_argument('hiddendim',
                       type=str
                      )
my_parser.add_argument('hiddenlay',
                       type=str
                      )
my_parser.add_argument('te',
                       type=str
                      )
my_parser.add_argument('avgemb',
                       type=str
                      )
my_parser.add_argument('useSkip',
                       type=str
                      )
my_parser.add_argument('lr',
                       type=str
                      )
my_parser.add_argument('useAux',
                       type=str
                      )
my_parser.add_argument('auxweight',
                       type=str
                      )
my_parser.add_argument('userembtype',
                       type=str
                      )

args = my_parser.parse_args()

learningrate = float(args.lr)
nfreq = int(args.nfreq)
useremb = int(args.useremb)
dropout = float(args.dropout)
grudim = int(args.grudim)
grulayers = int(args.grulayers)
hidden_lay = int(args.hiddenlay)
hidden_dim = int(args.hiddendim)
auxweight = float(args.auxweight)
userembtype = args.userembtype
if args.useAux=='useAux':
    useAux = True
else:
    useAux = False
if args.useSkip=='useSkip':
    useSkip = True
else:
    useSkip = False
if args.avgemb=='avgemb':
    useAvg = True
else:
    useAvg = False
if args.te=='TE':
    useTE = True
else:
    useTE = False
if args.grubi=='Bi':
    grubi = True
else:
    grubi = False
if args.pretrainedgru=='Pre':
    pretrained_gru = True
else:
    pretrained_gru = False
if args.sharedemb=='Share':
    shared_emb = True
else:
    shared_emb = False
if args.matrixfact=='Matrix':
    matrix_fact = True
else:
    matrix_fact = False
if args.altloss=='loss1':
    loss_weights = [2.0, 2.0, 5.0, 1.0] #reply, retweet, retweet_comment, like
elif args.altloss=='loss2':
    loss_weights = [5.0, 5.0, 10.0, 1.0] #reply, retweet, retweet_comment, like
print(nfreq)
print(useremb)
print(pretrained_gru)
print(shared_emb)
print(matrix_fact)
print(loss_weights)
print(dropout)
print(grudim)
print(grubi)
print(grulayers)
print(useTE)
print(useAvg)
print(learningrate)
print(auxweight)
print(userembtype)

dfuseremb=None
dfuseremba=None
dfuserembb=None

if userembtype=='Benny1':
    dfuseremb = pd.read_parquet('/raid/NN_encodings/user_id.parquet')
    dfuseremb = dfuseremb[dfuseremb['count']>nfreq]
    maxdim = dfuseremb['user_id_'].max()+2
elif userembtype=='Chris':
    dfuseremb = pd.read_parquet('/raid/abusercount.parquet')
    dfuseremb = dfuseremb[dfuseremb['count_b']>7]
    dfuseremb = dfuseremb[['user_id', 'count']].reset_index()
    dfuseremb.columns = ['drop', 'user_id', 'count']
    dfuseremb = dfuseremb.reset_index()
    dfuseremb = dfuseremb.drop(['drop'], axis=1)
    dfuseremb.columns = ['user_id_', 'user_id', 'count']
    dfuseremb['user_id_'] = dfuseremb['user_id_']+1
    maxdim = dfuseremb['user_id_'].max()+2
elif userembtype=='Benny2':
    dftmp = pd.read_parquet('/raid/abusercount_benny.parquet')
    dftmp = dftmp.reset_index()
    dftmp.columns = ['user_id_', 'a_user_id', 'count_a', 'b_user_id', 'count_b', 'count', 'user_id']
    dftmp = dftmp.reset_index()
    dftmp.columns = ['user_id_', 'drop', 'a_user_id', 'count_a', 'b_user_id', 'count_b', 'count', 'user_id']
    dftmp['user_id_'] = dftmp['user_id_']+2
    maxdim = dftmp['user_id_'].max()+2
    dfuserembb = dftmp[dftmp['count_b']>4][['user_id_', 'user_id', 'count']].copy().reset_index(drop=True)
    dfuseremba = dftmp[dftmp['count_a']>4][['user_id_', 'user_id', 'count']].copy().reset_index(drop=True)
    del dftmp
dfmuseremb = pd.read_parquet('/raid/NN_encodings/muser_id.parquet')
dfmuseremb = dfmuseremb[dfmuseremb['count']>9]
mmaxdim = dfmuseremb['muser_id_'].max()+2

dfhashtags = pd.read_parquet('/raid/NN_encodings/hashtags.parquet')
dfhashtags = dfhashtags[dfhashtags['count']>9]
hmaxdim = dfhashtags['hashtags_'].max()+2

dfdomains = pd.read_parquet('/raid/NN_encodings/domains.parquet')
dfdomains = dfdomains[dfdomains['count']>9]
dmaxdim = dfdomains['domains_'].max()+2

dfrtu = pd.read_parquet('/raid/NN_encodings/tw_rt_user0.parquet')
dfrtu = dfrtu[dfrtu['count']>9]
rtmaxdim = dfrtu['tw_rt_user0_'].max()+2

emb_shape['a_user_id_'] = [maxdim, useremb]
emb_shape['b_user_id_'] = [maxdim, useremb]
emb_shape['muser_id_'] = [mmaxdim, 16]
emb_shape['hashtags_'] = [hmaxdim, 16]
emb_shape['domains_'] = [dmaxdim, 16]
emb_shape['tw_rt_user0_'] = [rtmaxdim, 16]

print([maxdim, useremb])

if useTE:
    OTHERS_NUM = OTHERS_NUM + NUM_TE

valid_files = sorted(glob.glob('/raid/recsys2021_valid_pre_TEnn/*'))
print(valid_files)
df_valid = pd.concat([pd.read_parquet(x) for x in valid_files])
df_valid['id'] = np.asarray(range(df_valid.shape[0]))
quantiles = [240,  588, 1331, 3996]
df_valid['group'] = 0
for i, quant in enumerate(quantiles):
    df_valid['group'] = (df_valid['group']+(df_valid['a_follower_count']>quant).astype('int8')).astype('int8')
    
df_valid = preprocess_nn(df_valid, dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu, NUM_STATS, NUM_LOG_COLS, NUM_COLS, NUM_TE, useTE, userembtype, dfuseremba, dfuserembb)

print(df_valid['group'].value_counts())

df_valid = df_valid.sort_values('id').reset_index()
valid_dataset = CustomDataset(df_valid, max_len, OTHERS_NUM)
valid_loader = DataLoader(valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False, 
                          num_workers=16)

df_valid_group = df_valid[['group']].copy()
import gc
del df_valid
gc.collect()
modelfiles = sorted(glob.glob('/raid/nnmodels/*.pth'))
results = []

for modelfile in modelfiles:
    print(modelfile)
    start_time = time.time()
    simplemodel = torch.load(modelfile).module.cuda()
    scheduler=None
    #print(simplemodel)
    gc.collect()
    best_val_rce = -999999
    file_counter = 0
    label_names = ['reply', 'retweet', 'retweet_comment', 'like']
    valid_out = process_epoch(valid_loader, simplemodel, False, y_list=[], y_pred_list=[])
    valid_out_pred = np.vstack(valid_out[3]).copy()
    valid_out_gt = np.vstack(valid_out[2]).copy()
    rce_output = {}
    ap_output = {}
    for i, ind in enumerate([0,1,2,3]):
        print(ind)
        prauc_out = []
        rce_out = []
        ap_out = []
        for j in range(5):
            yvalid_tmp = valid_out_gt[df_valid_group['group']==j][:, i]
            oof_tmp = valid_out_pred[df_valid_group['group']==j][:, i]
            rce   = compute_rce_fast(cupy.asarray(oof_tmp), cupy.asarray(yvalid_tmp)).item()
            ap    = average_precision_score(yvalid_tmp, oof_tmp)
            rce_out.append(rce)
            ap_out.append(ap)
        rce_output[label_names[ind]] = rce_out
        ap_output[label_names[ind]] = ap_out

    valid_rce_sum = np.sum([np.mean(rce_output[label_name]) for label_name in label_names])
    valid_ap_sum = np.sum([np.mean(ap_output[label_name]) for label_name in label_names])
    total_time = time.time()-start_time
    tmp_result = {
        'modefile': modelfile,
        'valid_rce_sum': valid_rce_sum,
        'valid_rce_like': np.mean(rce_output['like']),
        'valid_rce_reply': np.mean(rce_output['reply']),
        'valid_rce_rt': np.mean(rce_output['retweet']),
        'valid_rce_rt_c': np.mean(rce_output['retweet_comment']),
        'valid_ap_sum': valid_ap_sum,
        'valid_ap_like': np.mean(ap_output['like']),
        'valid_ap_reply': np.mean(ap_output['reply']),
        'valid_ap_rt': np.mean(ap_output['retweet']),
        'valid_ap_rt_c': np.mean(ap_output['retweet_comment'])
    }
    print(total_time)
    print(tmp_result)
    results.append(tmp_result)
    pickle.dump(results, open('results_nn_23_val.pickle', 'wb'))
    del simplemodel
