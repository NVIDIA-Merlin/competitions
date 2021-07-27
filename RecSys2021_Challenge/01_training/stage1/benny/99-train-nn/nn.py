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

import torch
import torch.nn as nn

def get_emb_out(emb):
    out = 0
    for i in list(emb.embedding_layers):
        out+=i.embedding_dim
    return(out)

class ConcatenatedEmbeddings(torch.nn.Module):
    def __init__(self, col_list, emb_shape):
        super().__init__()
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(emb_shape[col][0], emb_shape[col][1])
                for col in col_list
            ]
        )

    def forward(self, x):
        x = [layer(x[:, i]) for i, layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, dim=1)
        return x
    
class MLPTower(nn.Module):
    def __init__(self, in_size, out_size, hidden_layers):
        super(MLPTower, self).__init__()
        layers = []
        for i, tmp_out_size in enumerate(hidden_layers):
            layers.append(nn.Linear(in_size, tmp_out_size))
            layers.append(nn.BatchNorm1d(tmp_out_size))
            layers.append(nn.ReLU())
            in_size = tmp_out_size
        layers.append(nn.Linear(in_size, out_size))
        self.hidden = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.hidden(X)

class AllNN(nn.Module):
    def __init__(self, 
                 wordemb=None,
                 gru_model=None,
                 berttiny_model=False,
                 berttiny=False,
                 pretrained_gru=False,
                 shared_emb=False,
                 matrix_fact=False,
                 hidden_dim=128,
                 hidden_lay=2,
                 TWEET_CAT=None,
                 TWEET_NUM=None,
                 A_USER_CAT=None,
                 A_USER_NUM=None,
                 B_USER_CAT=None,
                 B_USER_NUM=None,
                 OTHERS_CAT=None,
                 OTHERS_NUM=None,
                 emb_shape=None,
                 dropout=0.0,
                 GRU_DIM=64,
                 GRU_BI=False,
                 GRU_LAYERS=1,
                 useAvg=False,
                 useSkip=False,
                 auxloss=False
                ):
        super(AllNN, self).__init__()
        self.pretrained_gru=pretrained_gru
        self.shared_emb=shared_emb
        self.matrix_fact=matrix_fact
        self.berttiny = berttiny
        self.dropout = dropout
        self.ldropout = torch.nn.Dropout(dropout)
        self.useAvg = useAvg
        self.useSkip = useSkip
        self.auxloss = auxloss
        if self.berttiny:
            self.berttiny_model = berttiny_model
            textsize = 128
        else:
            textsize = 64
            if pretrained_gru:
                self.wordemb = gru_model.embblock
                self.gru = gru_model.lstm
            elif not useAvg:
                self.wordemb = wordemb
                self.gru = torch.nn.GRU(768, GRU_DIM, bidirectional=GRU_BI, batch_first=True, num_layers=GRU_LAYERS)
                textsize = GRU_DIM+int(GRU_DIM*GRU_BI)
            else:
                self.wordemb = nn.Embedding(119547, 256)
                textsize = 256
        self.musers_emb = ConcatenatedEmbeddings(['muser_id_'], emb_shape)
        self.tweet_emb = ConcatenatedEmbeddings(TWEET_CAT, emb_shape)
        self.tweet_num = nn.Linear(len(TWEET_NUM), len(TWEET_NUM))
        self.other_emb = ConcatenatedEmbeddings(OTHERS_CAT, emb_shape)
        self.other_num = nn.Linear(len(OTHERS_NUM), len(OTHERS_NUM))
        if shared_emb:
            self.user_emb = ConcatenatedEmbeddings(A_USER_CAT, emb_shape)
            self.user_num = nn.Linear(len(A_USER_NUM), len(A_USER_NUM))
            self.mlpuser = MLPTower(len(A_USER_NUM)+get_emb_out(self.user_emb), hidden_dim, [hidden_dim]*hidden_lay)
        else:
            self.a_user_emb = ConcatenatedEmbeddings(A_USER_CAT, emb_shape)
            self.b_user_emb = ConcatenatedEmbeddings(B_USER_CAT, emb_shape)
            self.a_user_num = nn.Linear(len(A_USER_NUM), len(A_USER_NUM))
            self.b_user_num = nn.Linear(len(B_USER_NUM), len(B_USER_NUM))
            self.mlpauser = MLPTower(len(A_USER_NUM)+get_emb_out(self.a_user_emb), hidden_dim, [hidden_dim]*hidden_lay)
            self.mlpbuser = MLPTower(len(B_USER_NUM)+get_emb_out(self.b_user_emb), hidden_dim, [hidden_dim]*hidden_lay)
        self.mlptweet = MLPTower(textsize+get_emb_out(self.tweet_emb)+get_emb_out(self.musers_emb)+len(TWEET_NUM), hidden_dim, [hidden_dim]*hidden_lay)
        hiddenother = max(8,(get_emb_out(self.other_emb)+len(OTHERS_NUM))//2)
        self.mlpother = MLPTower(get_emb_out(self.other_emb)+len(OTHERS_NUM), hiddenother, [hiddenother]*hidden_lay)
        if useSkip or auxloss:
            self.textskip = nn.Linear(textsize, 4)
            #self.auserskip = nn.Linear(hidden_dim, 4)
            self.buserskip = nn.Linear(hidden_dim, 4)
            self.otherskip = nn.Linear(hiddenother, 4)
        if useSkip:
            self.textskipw = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).float())
            #self.auserskipw = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).float())
            self.buserskipw = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).float())
            self.otherskipw = torch.nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5]).float())
        if matrix_fact:
            self.mlpfinal = MLPTower(hidden_dim+hiddenother, 4, [hidden_dim]*hidden_lay)
        else:
            self.mlpfinal = MLPTower(hidden_dim*2+hiddenother, 4, [hidden_dim]*hidden_lay)
        
        
        
    def forward(self, tokens, tweet_cat, tweet_num, a_user_cat, a_user_num, b_user_cat, b_user_num, other_cat, other_num, m_users_cat):
        if self.berttiny:
            xword = self.berttiny_model(tokens)['pooler_output']
        else:
            if not self.useAvg:
                xword = self.gru(self.wordemb(tokens))
                xword = xword[0][:,-1,:]
            else:
                tokenlist = []
                for ij in range(32):
                    tokenlist.append(torch.unsqueeze(self.wordemb(tokens[:, ij]), dim=1))
                xword = torch.cat(tokenlist, axis=1)
                xword = torch.mean(xword, axis=1)
        xmuser_list = []
        xmuser_cat = torch.sum(
            torch.cat(
                [torch.unsqueeze(
                    self.musers_emb(torch.unsqueeze(m_users_cat[:, i], dim=1)), 
                    dim=1) for i in range(3)], 
                axis=1), 
            axis=1
        )
        xtweet_cat = self.tweet_emb(tweet_cat)
        xtweet_num = self.tweet_num(tweet_num)
        xtweet = self.mlptweet(torch.cat([xword, xtweet_cat, xtweet_num, xmuser_cat], axis=1))
        if self.shared_emb:
            #xauser_cat = self.user_emb(a_user_cat)
            xbuser_cat = self.user_emb(b_user_cat)
            #xauser_num = self.user_num(a_user_num)
            xbuser_num = self.user_num(b_user_num)
            #xauser = self.mlpuser(torch.cat([xauser_cat, xauser_num], axis=1))
            xbuser = self.mlpuser(torch.cat([xbuser_cat, xbuser_num], axis=1))
        else:
            xauser_cat = self.a_user_emb(a_user_cat)
            xbuser_cat = self.b_user_emb(b_user_cat)
            xauser_num = self.a_user_num(a_user_num)
            xbuser_num = self.b_user_num(b_user_num)
            xauser = self.mlpauser(torch.cat([xauser_cat, xauser_num], axis=1))
            xbuser = self.mlpbuser(torch.cat([xbuser_cat, xbuser_num], axis=1))
        xother_cat = self.other_emb(other_cat)
        xother_num = self.other_num(other_num)
        xother = self.mlpother(torch.cat([xother_cat, xother_num], axis=1))
        if self.dropout>0:
            xtweet = self.ldropout(xtweet)
            xauser = self.ldropout(xauser)
            xbuser = self.ldropout(xbuser)
            xother = self.ldropout(xother)
        if self.matrix_fact:
            #xabuser = xauser*xbuser
            xtbuser = xtweet*xbuser
            xprefinal = torch.cat([xtbuser, xother], axis=1)
        else:
            xprefinal = torch.cat([xauser, xbuser, xtweet, xother], axis=1)
        xout = self.mlpfinal(xprefinal)
        auxout = []
        if self.useSkip or self.auxloss:
            xouttext = self.textskip(xword)
            #xoutauser = self.auserskip(xauser)
            xoutbuser = self.buserskip(xbuser)
            xoutother = self.otherskip(xother)
        if self.useSkip:
            xout = xout + self.textskipw*xouttext + self.buserskipw*xoutbuser + self.otherskipw*xoutother
        if self.auxloss:
            auxout.append(xouttext)
            #auxout.append(xoutauser)
            auxout.append(xoutbuser)
            auxout.append(xoutother)
        return(xout, auxout)

class SimpleNN(nn.Module):
    def __init__(self, 
                 embeddings,
                 lstm_hidden,
                 bi_di,
                 hidden_layers, 
                 EMB_DIM,
                 num_layers
                ):
        super(SimpleNN, self).__init__()
        self.embblock = embeddings
        self.lstm = torch.nn.GRU(EMB_DIM, lstm_hidden, bidirectional=bi_di, batch_first=True, num_layers=num_layers)
        self.topmlp = nn.Linear(lstm_hidden+lstm_hidden*bi_di, 4)
        
    def forward(self, tokens):
        x = self.embblock(tokens)
        #x = x.permute(1,0,2)
        x = self.lstm(x)
        x = x[0][:,-1,:]
        x_out = self.topmlp(x)
        return(x_out, x)