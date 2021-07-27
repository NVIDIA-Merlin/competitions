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
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
    

class ConcatenatedEmbeddings(torch.nn.Module):
    """Map multiple categorical variables to concatenated embeddings.
    Args:
        embedding_table_shapes: A dictionary mapping column names to
            (cardinality, embedding_size) tuples.
        dropout: A float.
    Inputs:
        x: An int64 Tensor with shape [batch_size, num_variables].
    Outputs:
        A Float Tensor with shape [batch_size, embedding_size_after_concat].
    """

    def __init__(self, embedding_table_shapes, dropout=0.0):
        super().__init__()
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(cat_size, emb_size) #, sparse=(cat_size > 1e5))
                for cat_size, emb_size in embedding_table_shapes.values()
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # first two cat columns (a_user and b_user) share same emb table            
        x = [self.embedding_layers[0](x[:,0])] + [layer(x[:, i+1]) for i, layer in enumerate(self.embedding_layers)] 
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return x
    
class Net(nn.Module):
    def __init__(self, num_features, layers, embedding_table_shapes, dropout=0.2, bert_type=None, gru_dim=128, emb_dim=768):
        super(Net, self).__init__()
        self.dropout = dropout
        self.initial_cat_layer = ConcatenatedEmbeddings(embedding_table_shapes, dropout=dropout)
        embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())
        layers = [layers] if type(layers) is int else layers
        layers = [num_features + gru_dim + embedding_size + 128 + 128] + layers
        self.use_bert = True
        # self.embed = AutoModel.from_pretrained(bert_type).embeddings.word_embeddings  
        self.embed = nn.Embedding(119547, emb_dim)
        assert emb_dim == self.embed.embedding_dim
#             self.reduce_dim = nn.Linear(self.embed.embedding_dim, 256)
#             self.embed = nn.Embedding(119547, emb_dim)
#         layers[0] += gru_dim
        self.lstm = nn.GRU(emb_dim, gru_dim, batch_first=True, bidirectional=False)    
#             self.lstm = nn.Linear(self.embed.embedding_dim, gru_dim)

        self.fn_layers = nn.ModuleList(
                            nn.Sequential(
                                nn.Dropout(p=dropout),
                                nn.Linear(layers[i], layers[i+1]),
                                nn.BatchNorm1d(layers[i+1]),
                                Swish_Module(),
                            )  for i in range(len(layers) -1)
                         )        
        self.fn_last = nn.Linear(layers[-1],4)
        
    def forward(self, x_cat, x_cont, bert_tok):
        a_emb = self.initial_cat_layer.embedding_layers[0](x_cat[:,0])
        b_emb = self.initial_cat_layer.embedding_layers[0](x_cat[:,1])
        mf = a_emb * b_emb        
        
        x_cat = self.initial_cat_layer(x_cat)
        bert_tok = self.embed(bert_tok)#.mean(dim=1)
#             bert_tok = self.reduce_dim(bert_tok)
        lstm_out = self.lstm(bert_tok)[0][:,-1]
        output = torch.cat([x_cont, lstm_out, x_cat, mf],dim=1)
        for layer in self.fn_layers:
            output = layer(output)
        logit = self.fn_last(output)
        return logit    

class AllDataset(Dataset):
    def __init__(self, df, tokens, max_len_txt, NUMERIC_COLUMNS, CAT_COLUMNS):
        self.X = df[NUMERIC_COLUMNS].values
        self.X_cat = df[CAT_COLUMNS].values
        self.text_tokens = tokens
        self.max_len_txt = max_len_txt
    def __len__(self):
        return self.text_tokens.shape[0]
    def __getitem__(self, index):        
        inputs = self.text_tokens[index][:self.max_len_txt]
        return self.X_cat[index], self.X[index].astype(np.float32), torch.tensor(inputs).long()
