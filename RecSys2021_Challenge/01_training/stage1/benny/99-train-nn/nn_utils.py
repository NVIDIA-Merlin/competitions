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
import cupy

import torch
import torch.nn as nn

from utils import *

class CustomDataset(Dataset):
    """Simple dataset class for dataloader"""
    def __init__(self, df, max_len, OTHERS_NUM1):
        """Initialize the CustomDataset"""
        self.tokens = np.vstack(df['text_tokens'].values)[:, :max_len]
        self.a_user_cat = df[A_USER_CAT].values
        self.b_user_cat = df[B_USER_CAT].values
        self.a_user_num = df[A_USER_NUM].values
        self.b_user_num = df[B_USER_NUM].values
        self.tweet_cat = df[TWEET_CAT].values
        self.tweet_num = df[TWEET_NUM].values
        self.other_cat = df[OTHERS_CAT].values
        self.other_num = df[OTHERS_NUM1].values
        self.target = df[TARGETS].values
        self.musers = df[['tw_original_user0_', 'tw_original_user1_', 'tw_original_user2_']].values
    
    def __len__(self):
        """Return the total length of the dataset"""
        dataset_size = self.tokens.shape[0]
        return dataset_size
  
    def __getitem__(self, idx):
        """Return the batch given the indices"""
        return (self.tokens[idx].astype(np.int64), 
                self.target[idx].astype(np.float32),
                self.a_user_cat[idx].astype(np.int64),
                self.b_user_cat[idx].astype(np.int64),
                self.tweet_cat[idx].astype(np.int64),
                self.other_cat[idx].astype(np.int64),
                self.a_user_num[idx].astype(np.float32),
                self.b_user_num[idx].astype(np.float32),
                self.tweet_num[idx].astype(np.float32),
                self.other_num[idx].astype(np.float32),
                self.musers[idx].astype(np.int64)
               )

def preprocess_nn(df, dfuseremb, dfmuseremb, dfhashtags, dfdomains, dfrtu, 
                  NUM_STATS, NUM_LOG_COLS, NUM_COLS, NUM_TE, useTE, userembtype, dfuseremba, dfuserembb):
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['date'].dt.hour
    df['a_ff_rate'] = (df['a_following_count'] / (1+df['a_follower_count'])).astype('float32')
    df['b_ff_rate'] = (df['b_follower_count']  / (1+df['b_following_count'])).astype('float32')
    df['ab_fing_rate'] = (df['a_following_count'] / (1+df['b_following_count'])).astype('float32')
    df['ab_fer_rate'] = (df['a_follower_count'] / (1+df['b_follower_count'])).astype('float32')
    df['ab_age_dff'] = (df['a_account_creation']-df['b_account_creation'])
    df['ab_age_rate'] = df['a_account_creation']/(1+df['b_account_creation'])
    df['tw_count_at'] = df['tw_count_at'].clip(0,6)
    df['tw_count_special1'] = df['tw_count_special1'].clip(0,6)
    df['tw_len_quest'] = df['tw_len_quest'].clip(0,6)
    df['tw_len_retweet'] = df['tw_len_retweet'].clip(0,1)
    df['tw_len_rt'] = df['tw_len_rt'].clip(0,1)
    if userembtype=='Benny2':
        col = 'a_user_id'
        dfuseremba.columns = [col + '_', col, 'count']
        df = df.merge(dfuseremba[[col + '_', col]], how='left', on=col)
        df[col + '_'] = df[col + '_'].fillna(0)
        col = 'b_user_id'
        dfuserembb.columns = [col + '_', col, 'count']
        df = df.merge(dfuserembb[[col + '_', col]], how='left', on=col)
        df[col + '_'] = df[col + '_'].fillna(1)
    else:
        col = 'a_user_id'
        dfuseremb.columns = [col + '_', col, 'count']
        df = df.merge(dfuseremb[[col + '_', col]], how='left', on=col)
        df[col + '_'] = df[col + '_'].fillna(0)
        col = 'b_user_id'
        dfuseremb.columns = [col + '_', col, 'count']
        df = df.merge(dfuseremb[[col + '_', col]], how='left', on=col)
        df[col + '_'] = df[col + '_'].fillna(0)
    col = 'hashtags'
    dfhashtags.columns = [col + '_', col, 'count']
    df = df.merge(dfhashtags[[col + '_', col]], how='left', on=col)
    df[col + '_'] = df[col + '_'].fillna(0)
    col = 'domains'
    dfdomains.columns = [col + '_', col, 'count']
    df = df.merge(dfdomains[[col + '_', col]], how='left', on=col)
    df[col + '_'] = df[col + '_'].fillna(0)
    col = 'tw_rt_user0'
    dfrtu.columns = [col + '_', col, 'count']
    df = df.merge(dfrtu[[col + '_', col]], how='left', on=col)
    df[col + '_'] = df[col + '_'].fillna(0)
    for col in ['tw_original_user0', 'tw_original_user1', 'tw_original_user2']:
        dfmuseremb.columns = [col + '_', col, 'count']
        df = df.merge(dfmuseremb[[col + '_', col]], how='left', on=col)
        df[col + '_'] = df[col + '_'].fillna(0)
    for col in NUM_LOG_COLS:
        df[col] = (np.log((df[col]+1).astype('float32'))-NUM_STATS[col][0])/NUM_STATS[col][1]
    for col in NUM_COLS:
        df[col] = ((df[col]).astype('float32')-NUM_STATS[col][0])/NUM_STATS[col][1]
    if useTE:
        for col in NUM_TE:
            df[col] = (df[col]-NUM_STATS[col][1])/(NUM_STATS[col][0]-NUM_STATS[col][1])
    return(df)    

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
    scheduler=None,
    auxloss=False,
    auxweight=1.0
):
    model.train(mode=train)
    for idx, batch in enumerate(iter(dataloader)):
        n+=batch[0].shape[0]
        tokens = batch[0].to('cuda')
        targets = batch[1].to('cuda')
        a_user_cat = batch[2].to('cuda')
        b_user_cat = batch[3].to('cuda')
        tweet_cat = batch[4].to('cuda')
        other_cat = batch[5].to('cuda')
        a_user_num = batch[6].to('cuda')
        b_user_num = batch[7].to('cuda')
        tweet_num = batch[8].to('cuda')
        other_num = batch[9].to('cuda')
        m_users_cat = batch[10].to('cuda')
        y = targets.float()
        nn_out = model(tokens, tweet_cat, tweet_num, a_user_cat, a_user_num, b_user_cat, b_user_num, other_cat, other_num, m_users_cat)
        y_pred = nn_out[0]
        loss0 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 0]), torch.squeeze(y[:, 0]))
        loss1 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 1]), torch.squeeze(y[:, 1]))
        loss2 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 2]), torch.squeeze(y[:, 2]))
        loss3 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 3]), torch.squeeze(y[:, 3]))
        loss = loss_weights[0]*loss0 + loss_weights[1]*loss1 + loss_weights[2]*loss2 + loss_weights[3]*loss3
        if auxloss:
            lossaux = 0
            for y_pred_aux in nn_out[1]:
                aloss0 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 0]), torch.squeeze(y[:, 0]))
                aloss1 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 1]), torch.squeeze(y[:, 1]))
                aloss2 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 2]), torch.squeeze(y[:, 2]))
                aloss3 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 3]), torch.squeeze(y[:, 3]))
                lossaux = lossaux + loss_weights[0]*aloss0 + loss_weights[1]*aloss1 + loss_weights[2]*aloss2 + loss_weights[3]*aloss3
            loss = loss + auxweight*lossaux
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        y_list.append(y.detach().cpu().numpy())
        y_pred_list.append(torch.sigmoid(y_pred).detach().cpu().numpy())
        total_loss += loss.detach().cpu().item()*n
    return total_loss, n, y_list, y_pred_list

def process_epoch_pred(
    dataloader,
    model,
    train=False,
    optimizer=None,
    total_loss=0.0,
    n=0,
    y_list=[],
    y_pred_list=[],
    loss_weights=[1.0,1.0,1.0,1.0],
    scheduler=None,
    auxloss=False,
    auxweight=1.0
):
    auxloss_total = []
    model.train(mode=train)
    for idx, batch in enumerate(iter(dataloader)):
        n+=batch[0].shape[0]
        tokens = batch[0].to('cuda')
        targets = batch[1].to('cuda')
        a_user_cat = batch[2].to('cuda')
        b_user_cat = batch[3].to('cuda')
        tweet_cat = batch[4].to('cuda')
        other_cat = batch[5].to('cuda')
        a_user_num = batch[6].to('cuda')
        b_user_num = batch[7].to('cuda')
        tweet_num = batch[8].to('cuda')
        other_num = batch[9].to('cuda')
        m_users_cat = batch[10].to('cuda')
        y = targets.float()
        nn_out = model(tokens, tweet_cat, tweet_num, a_user_cat, a_user_num, b_user_cat, b_user_num, other_cat, other_num, m_users_cat)
        y_pred = nn_out[0]
        loss0 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 0]), torch.squeeze(y[:, 0]))
        loss1 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 1]), torch.squeeze(y[:, 1]))
        loss2 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 2]), torch.squeeze(y[:, 2]))
        loss3 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred[:, 3]), torch.squeeze(y[:, 3]))
        loss = loss_weights[0]*loss0 + loss_weights[1]*loss1 + loss_weights[2]*loss2 + loss_weights[3]*loss3
        if auxloss:
            lossaux = 0
            for y_pred_aux in nn_out[1]:
                aloss0 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 0]), torch.squeeze(y[:, 0]))
                aloss1 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 1]), torch.squeeze(y[:, 1]))
                aloss2 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 2]), torch.squeeze(y[:, 2]))
                aloss3 = torch.nn.BCEWithLogitsLoss()(torch.squeeze(y_pred_aux[:, 3]), torch.squeeze(y[:, 3]))
                lossaux = lossaux + loss_weights[0]*aloss0 + loss_weights[1]*aloss1 + loss_weights[2]*aloss2 + loss_weights[3]*aloss3
            loss = loss + auxweight*lossaux
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        y_list.append(y.detach().cpu().numpy())
        y_pred_list.append(torch.sigmoid(y_pred).detach().cpu().numpy())
        auxloss_total_tmp = []
        for y_pred_auxloss in nn_out[1]:
            auxloss_total_tmp.append(torch.sigmoid(y_pred_auxloss).detach().cpu().numpy())
        auxloss_total.append(auxloss_total_tmp)
        total_loss += loss.detach().cpu().item()*n
    return total_loss, n, y_list, y_pred_list, auxloss_total

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

def display_score(rce,ap):
    print('Quantile Group|AP Retweet|RCE Retweet|  AP Reply|  RCE Reply|   AP Like|   RCE Like|AP RT comment|RCE RT comment')
    for i in range(5):
        print(f'{i:9}      ' + \
              ' '.join([f"{ap[engage_type][i]:10.4f}  {rce[engage_type][i]:10.4f}" for engage_type in ['retweet','reply','like','retweet_comment']]))

    print('     Average   ' + ' '.join([f"{np.mean(list(ap[engage_type])):10.4f}  {np.mean(list(rce[engage_type])):10.4f}" for engage_type in ['retweet','reply','like','retweet_comment']]))            

def precision_recall_curve(y_true,y_pred):
    y_true = y_true.astype('float32')
    ids = cupy.argsort(-y_pred) 
    y_true = y_true[ids]
    y_pred = y_pred[ids]
    y_pred = cupy.flip(y_pred,axis=0)

    acc_one = cupy.cumsum(y_true)
    sum_one = cupy.sum(y_true)
    
    precision = cupy.flip(acc_one/cupy.cumsum(cupy.ones(len(y_true))),axis=0)
    precision[:-1] = precision[1:]
    precision[-1] = 1.

    recall = cupy.flip(acc_one/sum_one,axis=0)
    recall[:-1] = recall[1:]
    recall[-1] = 0
    n = (recall==1).sum()
    
    return precision[n-1:],recall[n-1:],y_pred[n:]

def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    recall, prec = cupy.asnumpy(recall), cupy.asnumpy(prec)
    prauc = auc(recall, prec)
    return prauc

def log_loss_cp(y_true,y_pred,eps=1e-15, normalize=True, sample_weight=None):
    y_true = y_true.astype('int32')
    y_pred = cupy.clip(y_pred, eps, 1 - eps)
    if y_pred.ndim == 1:
        y_pred = cupy.expand_dims(y_pred, axis=1)
    if y_pred.shape[1] == 1:
        y_pred = cupy.hstack([1 - y_pred, y_pred])

    y_pred /= cupy.sum(y_pred, axis=1, keepdims=True)
    loss = -cupy.log(y_pred)[cupy.arange(y_pred.shape[0]), y_true]
    return _weighted_sum(loss, sample_weight, normalize).item()

def _weighted_sum(sample_score, sample_weight, normalize):
    if normalize:
        return cupy.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return cupy.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()

# FAST METRIC FROM GIBA
def compute_rce_fast(pred, gt):
    cross_entropy = log_loss_cp(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

from sklearn.metrics import average_precision_score, log_loss

def calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0