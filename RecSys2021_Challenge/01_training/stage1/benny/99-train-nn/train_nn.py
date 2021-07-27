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

import pickle

from nn import AllNN
from nn_utils import CustomDataset, compute_rce, process_epoch, preprocess_nn
from utils import *

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import argparse

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
berttiny_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
berttiny_model.resize_token_embeddings(119547)
berttiny_model.init_weights()

model = BertModel.from_pretrained("bert-base-multilingual-cased")
wordemb = model.embeddings.word_embeddings
grumodel = torch.load('model_best_2.pth')

DEBUG = False
loadcheckpoint = False

nfreq = 15
BATCH_SIZE = 1024*4
pretrained_gru = True
berttiny=False
shared_emb=False
matrix_fact=False
hidden_dim=64
hidden_lay=2
max_len = 42
EPOCHS = 2
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

if not loadcheckpoint:
    simplemodel = AllNN(wordemb, 
                        grumodel, 
                        pretrained_gru=pretrained_gru,
                        berttiny_model=berttiny_model,
                        berttiny=berttiny,
                        shared_emb=shared_emb,
                        matrix_fact=matrix_fact,
                        hidden_dim=hidden_dim,
                        hidden_lay=hidden_lay,
                        TWEET_CAT=TWEET_CAT,
                        TWEET_NUM=TWEET_NUM,
                        A_USER_CAT=A_USER_CAT,
                        A_USER_NUM=A_USER_NUM,
                        B_USER_CAT=B_USER_CAT,
                        B_USER_NUM=B_USER_NUM,
                        OTHERS_CAT=OTHERS_CAT,
                        OTHERS_NUM=OTHERS_NUM,
                        emb_shape=emb_shape, 
                        dropout=dropout,
                        GRU_DIM=grudim,
                        GRU_BI=grubi,
                        GRU_LAYERS=grulayers, 
                        useAvg=useAvg, 
                        useSkip=useSkip, 
                        auxloss=useAux).cuda()
else:
    simplemodel = torch.load('/results/models_emb_week12/model_last.pth').module
if pretrained_gru:
    parameters = simplemodel.parameters()
else:
    simplemodel.wordemb.weight.requires_grad=False
    parameters = filter(lambda p: p.requires_grad, simplemodel.parameters())
simplemodel = torch.nn.DataParallel(simplemodel, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda() 
    

if optimizer_type=='Adam':
    optimizer = torch.optim.Adam(parameters, lr=learningrate)
if optimizer_type=='AdamW':
    optimizer = torch.optim.AdamW(parameters, lr=learningrate)

# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
#                                               base_lr=0.0001, 
#                                               max_lr=learningrate, 
#                                               mode='triangular2',
#                                               step_size_up=50000
#                                              )
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
#                                                        eta_min = 0.0001,
#                                                        T_max=50000
#                                                       )
scheduler=None
print(simplemodel)

train_files = sorted(glob.glob('/raid/recsys2021_pre_1/*')) + sorted(glob.glob('/raid/recsys2021_pre_2/*'))
valid_files = sorted(glob.glob('/raid/recsys2021_pre_3_valid/*'))[0:20]

if DEBUG:
    print(valid_files[0])
    df_valid = pd.read_parquet(valid_files[0]).head(10000)
    df_valid['id'] = np.asarray(range(df_valid.shape[0]))
else:
    df_valid = pd.concat([pd.read_parquet(x) for x in valid_files])
    df_valid['id'] = np.asarray(range(df_valid.shape[0]))

df_valid = preprocess_nn(df_valid, dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu, NUM_STATS, NUM_LOG_COLS, NUM_COLS, NUM_TE, useTE, userembtype, dfuseremba, dfuserembb)
df_valid = df_valid.sort_values('id').reset_index()
valid_dataset = CustomDataset(df_valid, max_len, OTHERS_NUM)
valid_loader = DataLoader(valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False, 
                          num_workers=16)

print(df_valid['a_user_id_'].max())
print(df_valid['b_user_id_'].max())
gc.collect()
results = []
best_val_rce = -999999
file_counter = 0
label_names = ['reply', 'retweet', 'retweet_comment', 'like']

for e in list(range(EPOCHS)):
    random.shuffle(train_files)
    for ifile, train_file in enumerate(train_files):
        start_time = time.time()
        print(train_file)
        if DEBUG:
            df = pd.read_parquet(train_file).head(10000)
        else:
            df = pd.read_parquet(train_file)
        df = preprocess_nn(df, dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu, NUM_STATS, NUM_LOG_COLS, NUM_COLS, NUM_TE, useTE, userembtype, dfuseremba, dfuserembb)
        train_dataset = CustomDataset(df, max_len, OTHERS_NUM)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=16, drop_last=True)
        train_out = process_epoch(train_loader, 
                                  simplemodel, 
                                  True, 
                                  optimizer, 
                                  y_list=[], 
                                  y_pred_list=[], 
                                  loss_weights=loss_weights, 
                                  scheduler=scheduler, 
                                  auxloss=useAux, 
                                  auxweight=auxweight)
        file_counter+=1
        print(time.time()-start_time)
        if ((file_counter%25)==0):
            torch.save(simplemodel, '/raid/model_step_' + str(file_counter) + '.pth')
            os.system('screen -d -m cp /raid/model_step_' + str(file_counter) + '.pth /results/')  
        if ((file_counter%10)==0) or (file_counter==1):
            valid_out = process_epoch(valid_loader, simplemodel, False, y_list=[], y_pred_list=[])
            valid_out_pred = np.vstack(valid_out[3]).copy()
            valid_out_gt = np.vstack(valid_out[2]).copy()
            rce_output = {}
            ap_output = {}
            for i, ind in enumerate([0,1,2,3]):
                prauc_out = []
                rce_out = []
                ap_out = []
                for j in range(5):
                    yvalid_tmp = valid_out_gt[df_valid['group']==j][:, i]
                    oof_tmp = valid_out_pred[df_valid['group']==j][:, i]
                    rce   = compute_rce(oof_tmp, yvalid_tmp)
                    ap    = 0
                    rce_out.append(rce)
                    ap_out.append(ap)
                rce_output[label_names[ind]] = rce_out
                ap_output[label_names[ind]] = ap_out

            valid_rce_sum = np.sum([np.mean(rce_output[label_name]) for label_name in label_names])
            total_time = time.time()-start_time
            tmp_result = {
                'epoch': e,
                'ifile': ifile,
                'time': total_time,
                'valid_rce_sum': valid_rce_sum,
                'valid_rce_like': np.mean(rce_output['like']),
                'valid_rce_reply': np.mean(rce_output['reply']),
                'valid_rce_rt': np.mean(rce_output['retweet']),
                'valid_rce_rt_c': np.mean(rce_output['retweet_comment'])
            }
            filepost = str(nfreq) + '_' + str(useremb) + '_' + str(pretrained_gru) + '_' + str(shared_emb) + '_' + str(matrix_fact) + '_' + str(loss_weights) + '_' + str(dropout) + '_' + str(grudim) + '_' + str(grubi) + '_' + str(grulayers) + '_' + str(hidden_dim) + '_' + str(hidden_lay) + '_' + str(useTE) + '_' + str(useAvg) + '_' + str(useSkip) + '_' + str(learningrate) + '_' + str(useAux) + '_' + str(auxweight) + '_' + str(userembtype)
            torch.save(simplemodel, '/raid/model_last.pth')
            if valid_rce_sum>best_val_rce:
                print('Save best model')
                torch.save(simplemodel, '/raid/model_best.pth')
                best_val_rce = valid_rce_sum
                pickle.dump([valid_out_pred, valid_out_gt], open('./pred/pred_' + filepost  + '.pickle', 'wb'))
            print(tmp_result)
            results.append(tmp_result)
            if useSkip:
                print(simplemodel.textskipw)
                print(simplemodel.auserskipw)
                print(simplemodel.buserskipw)
                print(simplemodel.otherskipw)
            pickle.dump(results, open('results_nn_2_' + filepost  + '.pickle', 'wb'))
            os.system('screen -d -m cp /raid/model_best.pth /results/')
            os.system('screen -d -m cp /raid/model_last.pth /results/')

#         def df
#         gc.collect()

os.system('cp /raid/model*.pth /results/')