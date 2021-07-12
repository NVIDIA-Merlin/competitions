#
# The MIT License (MIT)

# Copyright (c) 2021, NVIDIA CORPORATION

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import cudf
import cupy
import nvtabular as nvt
import json
from pathlib import *
import glob
from functools import partial
from scipy.spatial.distance import cosine

def generate_xgb_feats(data, search_session, mapping, image_matrix, desc_matrix, mapping_id_sku_emb_position): 
    
    # load search data and map product skus 
    # mapping = pd.read_parquet(os.path.join(args.data_path,
    #'categorify_workflow/categories/unique.product_url_hash_first_purchase_id_first_AC_id.parquet'))
    mapping_dict = dict(zip(mapping.product_url_hash_first_purchase_id_first_AC_id, mapping.index))
    # Update mapping with unseen browsing products ids present in flat_product_skus_hash and flat_clicked_skus_hash
    prods = [e for impression in search_session.flat_product_skus_hash.values for e in impression]
    clicked_prod = [e for impression in search_session.flat_clicked_skus_hash.values for e in impression]
    all_prods = set(prods + clicked_prod)
    new_prods = all_prods.difference(set(mapping.product_url_hash_first_purchase_id_first_AC_id))
    print('Number of products present in search and not in browsing data is: %s' %len(new_prods))
    
    cardinality = len(mapping_dict)
    new_ids = list(range(cardinality, cardinality+len(new_prods)))
    new_dict = dict(zip(new_prods, new_ids))
    mapping_dict.update(new_dict)
    search_session.flat_product_skus_hash = search_session.flat_product_skus_hash.progress_apply(lambda x:  [mapping_dict[e] for e in x])
    search_session.flat_clicked_skus_hash = search_session.flat_clicked_skus_hash.progress_apply(lambda x:  [mapping_dict[e] for e in x])
    
    
    
    add_to_cart_data = data.progress_apply(add_to_cart_features, axis=1)
    session_data = data.progress_apply(session_feartures, axis=1)
    p_last_data = data.progress_apply(get_p_last_interactions, axis=1)
    similarity_data = data.progress_apply(partial(compute_similarity, 
                                                  image_matrix=image_matrix,
                                                  desc_matrix=desc_matrix,
                                                 mapping_id_sku_emb_position=mapping_id_sku_emb_position), axis=1)
    

    
    add_to_cart_data = pd.DataFrame(add_to_cart_data.tolist(), columns=[
                                              'add_product_id',
                                              'add_nb_interactions',
                                              'add_has_been_detailed',
                                              'add_has_been_removed',
                                              'add_has_been_searched',
                                              'add_has_been_clicked',
                                              'add_category_hash',
                                              'add_price', 
                                              'add_relative_price'])

    session_data = pd.DataFrame(session_data.tolist(), columns=[
                                              'session_length', 'nb_before_add',
                                              'nb_unique_interactions', 'nb_queries'])

    similarity_data = pd.DataFrame(similarity_data.tolist(), columns=[ 
                                                                     'mean_sim_desc', 'std_sim_desc', 
                                                                      'mean_sim_img', 'std_sim_img'])


    p_last_data = pd.DataFrame(p_last_data.tolist(), columns=[
                                              'product_url_id_list',                 
                                              'event_type_list',
                                              'product_action_list',
                                              'category_list', 
                                              'price_list', 
                                              'relative_price_list'])

    p = 5
    # convert list column to multiple columns
    for col in ['product_url_id_list',   'event_type_list',
                'product_action_list', 'category_list', 
                 'price_list', 'relative_price_list']: 
        t = pd.DataFrame(p_last_data[col].to_list(), columns=[col+"-%s"%i for i in range(p)])
        p_last_data = pd.concat([p_last_data, t], axis=1)
        p_last_data.drop(col, axis=1, inplace=True)
    
    # fill missing : replace missing values of categorical by '0'
    for col in ['product_url_id_list',   'event_type_list',
                'product_action_list', 'category_list', 
                 'price_list']: 
        for sub_col in [col+"-%s"%i for i in range(p)]: 
            p_last_data[sub_col] = p_last_data[sub_col].fillna(0)
            
    # fill missing : replace missing values of continuous by the mediane '-0.000199'
    for sub_col in ['relative_price_list'+"-%s"%i for i in range(p)]: 
        p_last_data[sub_col] = p_last_data[sub_col].fillna(-0.000199)
    
    data.reset_index(drop=True, inplace=True),
    p_last_data.reset_index(drop=True, inplace=True), 
    add_to_cart_data.reset_index(drop=True, inplace=True),
    session_data.reset_index(drop=True, inplace=True)
                
    
    xgboost_frame = pd.concat([data[['original_session_id_hash', 'session_id_hash', 
                                     'is_purchased-last', 'nb_after_add-last',
                                     'is_test-last', 'is_valid', 'fold']],
                               p_last_data, 
                               add_to_cart_data,
                               session_data], axis=1)
    
    # fill missing with zeros 
    xgboost_frame = xgboost_frame.fillna(0)
    return xgboost_frame



def compute_similarity(x, image_matrix, desc_matrix, mapping_id_sku_emb_position): 
    product_id = x['add_product_id']
    
    unique_product = list(set(x['product_url_hash_list']))
    
    positions = [mapping_id_sku_emb_position[id] for id in unique_product if id in mapping_id_sku_emb_position]
    
    interaction_desc = desc_matrix[positions]
    interaction_img = image_matrix[positions]
    
    add_interaction_desc = desc_matrix[mapping_id_sku_emb_position[product_id]]
    add_interaction_img = image_matrix[mapping_id_sku_emb_position[product_id]]
    
    similarities_desc = [cosine(add_interaction_desc, b) for b in interaction_desc]
    similarities_img = [cosine(add_interaction_img, b) for b in interaction_img]
    
    return [np.mean(similarities_desc), np.std(similarities_desc), 
            np.mean(similarities_img), np.std(similarities_img)]


def add_to_cart_features(x):
    add_index =  x['product_action-list'].tolist().index(1)
    product_id = x['product_url_hash_list'][add_index]
    product_interactions_index = np.where(x['product_url_hash_list'] == product_id)[0]
    actions = x['product_action-list'][product_interactions_index]
    nb_interactions = len(actions)
    has_been_detailed = 3 in list(actions)
    has_been_removed = 5 in list(actions)
    has_been_clicked = 2 in list(actions)
    if type(x['flat_clicked_skus_hash']) is not float:
        has_been_searched = product_id in x['flat_product_skus_hash']
    else: 
        has_been_searched = 0
    product_category = x['category_hash-list'][add_index]
    product_price = x['price_bucket-list'][add_index]
    product_relative_category_price = x['mean_price_hierarchy-list'][add_index]
    return ( product_id, nb_interactions, has_been_detailed, has_been_removed, 
            has_been_searched, has_been_clicked, product_category,
            product_price, product_relative_category_price)

def session_feartures(x): 
    session_len = len(x['product_url_hash_list'])
    unique_items =  len(set(x['product_url_hash_list']))
    add_index =  x['product_action-list'].tolist().index(1)
    nb_before_add = len(x['product_url_hash_list'][:add_index])
    return [session_len, nb_before_add, unique_items, x['nb_queries']]

def get_p_last_interactions(x, p=5): 
    cols = ['product_url_hash_list', 'event_type-list', 'product_action-list', 'category_hash-list',
           'price_bucket-list', 'mean_price_hierarchy-list']
    last_p_interactions  = [x[col][-p:] for col in cols]
    return  last_p_interactions
