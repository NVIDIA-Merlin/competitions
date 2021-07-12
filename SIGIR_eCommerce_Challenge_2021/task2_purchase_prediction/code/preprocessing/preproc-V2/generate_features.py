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

from functools import partial 
import glob
from scipy.spatial.distance import cosine
import pickle
import numpy as np 
import pandas as pd 


def generate_xgb_feats(data, search_session, mapping, image_matrix, desc_matrix, mapping_id_sku_emb_position): 
    # load search data and map product skus 
    # mapping = pd.read_parquet(os.path.join(args.data_path, 'categorify_workflow/categories/unique.product_url_hash_first_purchase_id_first_AC_id.parquet'))
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
    
    # merge search and session columns
    data = data.merge(search_session, on='original_session_id_hash', how='left')
    
    # generate xgb features: 90 in total 
    print("generate before/after count statistics")
    statisitcs_bef_aft_AC = data.progress_apply(statistics_before_after_AC, axis=1)
    print("generate add-to-cart (AC) features")
    add_to_cart_data = data.progress_apply(add_to_cart_features, axis=1)
    print("generate session-level features")
    session_data = data.progress_apply(session_feartures, axis=1)
    print("generate p first/last interactions")
    p_interactions_data = data.progress_apply(get_p_interactions, axis=1)
    print("generate similarity features")
    similarity_data = data.progress_apply(partial(compute_similarity, 
                                                  image_matrix=image_matrix,
                                                  desc_matrix=desc_matrix,
                                                 mapping_id_sku_emb_position=mapping_id_sku_emb_position), axis=1)
    
    
    # Build Data-Frames 
    statisitcs_bef_aft_AC_data = pd.DataFrame(statisitcs_bef_aft_AC.tolist(), columns=['nb_add_before', 'nb_add_after',
                                                                                      'nb_detail_before','nb_detail_after',
                                                                                      'nb_remove_before', 'nb_remove_after',
                                                                                      'nb_view_before', 'nb_view_after' ,
                                                                                      'nb_click_before', 'nb_click_after'])
    

    add_to_cart_data = pd.DataFrame(add_to_cart_data.tolist(), columns=[ 'add_product_id', 'add_nb_interactions',
                                                                          'add_has_been_detailed','add_has_been_removed',
                                                                          'add_has_been_viewed','add_has_been_searched',
                                                                          'add_has_been_clicked', 'add_category_hash',
                                                                          'add_main_category', 'add_price', 
                                                                          'add_relative_price',  'add_relative_price_main' ])


    session_data = pd.DataFrame(session_data.tolist(), columns=[     
                                              'session_length',
                                              'nb_unique_interactions',
                                              'nb_queries'])


    similarity_data = pd.DataFrame(similarity_data.tolist(), columns=[
                                                                      'mean_sim_desc', 'std_sim_desc', 
                                                                      'mean_sim_img', 'std_sim_img',
                                                                      'mean_sim_desc_before', 'std_sim_desc_before', 
                                                                      'mean_sim_img_before', 'std_sim_img_before',
                                                                      'mean_sim_desc_after', 'std_sim_desc_after', 
                                                                      'mean_sim_img_after', 'std_sim_img_after', 
                                                                      'main_category_similarity_general', 'main_category_similarity_add'
                                                                     ])



    p_last_data = pd.DataFrame(p_interactions_data.tolist(), columns=[
                                              'product_url_id_list_after',                 
                                              'event_type_list_after',
                                              'product_action_list_after',
                                              'category_list_after', 
                                              'price_list_after', 
                                              'relative_price_list_after',
                                              'product_url_id_list_before',                 
                                              'event_type_list_before',
                                              'product_action_list_before',
                                              'category_list_before', 
                                              'price_list_before', 
                                              'relative_price_list_before'
                                                             ])
    p = 5
    print("Create columns of first and last interactions")
    # convert list column to multiple columns
    for col in ['product_url_id_list_after',    'event_type_list_after',
                'product_action_list_after',  'category_list_after', 
                 'price_list_after', 'relative_price_list_after',

                'product_url_id_list_before',  'event_type_list_before',
                'product_action_list_before', 'category_list_before', 
                'price_list_before',   'relative_price_list_before']: 
        t = pd.DataFrame(p_last_data[col].to_list(), columns=[col+"-%s"%i for i in range(p)])
        p_last_data = pd.concat([p_last_data, t], axis=1)
        p_last_data.drop(col, axis=1, inplace=True)
        
    del t 
    # fill missing : replace missing values of categorical by '0'
    for col in ['product_url_id_list_after',    'event_type_list_after',
                'product_action_list_after',  'category_list_after', 
                'price_list_after', 'relative_price_list_after',
                'product_url_id_list_before',  'event_type_list_before',
                'product_action_list_before', 'category_list_before', 
                'price_list_before',   ]: 
        for sub_col in [col+"-%s"%i for i in range(p)]: 
            p_last_data[sub_col] = p_last_data[sub_col].fillna(0)
    # fill missing : replace missing values of continuous by the mediane '-0.000199'
    for sub_col in ['relative_price_list_before'+"-%s"%i for i in range(p)]+ ['relative_price_list_after'+"-%s"%i for i in range(p)] : 
        p_last_data[sub_col] = p_last_data[sub_col].fillna(0.028996615)
    
    
    # merge data-frame
    print("Merge all features")
    xgboost_frame = pd.concat([data[['original_session_id_hash', 'session_id_hash', 
                                     'is_purchased-last', 'nb_after_add-last',
                                     'is_test-last', 'is_valid', 'fold']], p_last_data, add_to_cart_data,
                               session_data, statisitcs_bef_aft_AC_data, similarity_data], axis=1)

    del add_to_cart_data
    del session_data
    del statisitcs_bef_aft_AC_data
    del data
    del similarity_data
    
    # fill missing with zeros 
    xgboost_frame = xgboost_frame.fillna(0)
    return xgboost_frame


    
def statistics_before_after_AC(x): 
    add_index = x['product_url_hash_list'].tolist().index(x['first_AC_id-last'])
    nb_add_before,  nb_add_after = x['has_been_added_to_cart-list'][:add_index].sum(), x['has_been_added_to_cart-list'][add_index+1:].sum() 
    nb_detail_before,  nb_detail_after = x['has_been_detailed-list'][:add_index].sum(), x['has_been_detailed-list'][add_index+1:].sum() 
    nb_remove_before,  nb_remove_after = x['has_been_removed_from_cart-list'][:add_index].sum(), x['has_been_removed_from_cart-list'][add_index+1:].sum() 
    nb_click_before,  nb_click_after = x['has_been_clicked-list'][:add_index].sum(), x['has_been_clicked-list'][add_index+1:].sum() 
    nb_view_before,  nb_view_after = x['has_been_viewed-list'][:add_index].sum(), x['has_been_viewed-list'][add_index+1:].sum() 
    
    return [ nb_add_before,  nb_add_after, 
              nb_detail_before,  nb_detail_after,
              nb_remove_before,  nb_remove_after,
              nb_view_before,  nb_view_after,
              nb_click_before,  nb_click_after]


def compute_similarity(x, desc_matrix, image_matrix, mapping_id_sku_emb_position): 
    # compute similarity vectors between add-to-cart product and beforei interactions, after interactions 
    add_product_id = x['first_AC_id-last']
    add_index = x['product_url_hash_list'].tolist().index(add_product_id)
    add_interaction_desc = desc_matrix[mapping_id_sku_emb_position[add_product_id]]
    add_interaction_img = image_matrix[mapping_id_sku_emb_position[add_product_id]]
    
    unique_product = list(set(x['product_url_hash_list']))
    unique_product.remove(add_product_id)
    unique_product_before = list(set(x['product_url_hash_list'][:add_index]))
    unique_product_after = list(set(x['product_url_hash_list'][add_index+1:]))
    
    if unique_product is not None:
        positions = [mapping_id_sku_emb_position[id] for id in unique_product if id in mapping_id_sku_emb_position]
        interaction_desc = desc_matrix[positions]
        interaction_img = image_matrix[positions]
        similarities_desc = [cosine(add_interaction_desc, b) for b in interaction_desc]
        similarities_img = [cosine(add_interaction_img, b) for b in interaction_img]   
    else: 
        # only product_id of AC in the session, sim is 1 
        similarities_desc, similarities_img = [1], [1]
        
        
    if unique_product_before is not None:
        positions_before = [mapping_id_sku_emb_position[id] for id in unique_product_before if id in mapping_id_sku_emb_position]
        interaction_desc_before = desc_matrix[positions_before]
        interaction_img_before = image_matrix[positions_before]
        similarities_desc_before = [cosine(add_interaction_desc, b) for b in interaction_desc_before]
        similarities_img_before = [cosine(add_interaction_img, b) for b in interaction_img_before]
    else: 
        similarities_desc_before, similarities_img_before = [0], [0]
    
    
    if unique_product_after is not None:
        positions_after = [mapping_id_sku_emb_position[id] for id in unique_product_after if id in mapping_id_sku_emb_position]
        interaction_desc_after = desc_matrix[positions_after]
        interaction_img_after = image_matrix[positions_after]
        similarities_desc_after = [cosine(add_interaction_desc, b) for b in interaction_desc_after]
        similarities_img_after = [cosine(add_interaction_img, b) for b in interaction_img_after]
    else: 
        similarities_desc_after, similarities_img_after = [0], [0]
    
    main_category_similarity_general = len(set(x['main_category-list'])) / len(x['main_category-list'])
    
    add_category = x['main_category-list'][add_index]
    main_category_similarity_add = np.sum(x['main_category-list'] == add_category) / len(x['main_category-list'])
    
    
    
    return [ np.mean(similarities_desc), np.std(similarities_desc), 
            np.mean(similarities_img), np.std(similarities_img),
            np.mean(similarities_desc_before), np.std(similarities_desc_before), 
            np.mean(similarities_img_before), np.std(similarities_img_before),
            np.mean(similarities_desc_after), np.std(similarities_desc_after), 
            np.mean(similarities_img_after), np.std(similarities_img_after),
            main_category_similarity_general, main_category_similarity_add
           ]


def add_to_cart_features(x):
    add_product_id = x['first_AC_id-last']
    add_index = x['product_url_hash_list'].tolist().index(add_product_id)
    
    add_nb_interactions = x['product_nb_interactions-list'][add_index]
    has_been_detailed = x['has_been_detailed-list'][add_index]
    has_been_removed = x['has_been_removed_from_cart-list'][add_index]
    has_been_viewed = x['has_been_viewed-list'][add_index]
    has_been_clicked = x['has_been_clicked-list'][add_index]
    if type(x['flat_product_skus_hash']) is not float:
        has_been_searched = add_product_id in x['flat_product_skus_hash']
    has_been_searched = 0
    
    product_category = x['category_hash-list'][add_index]
    product_main_category = x['main_category-list'][add_index]
    product_price = x['price_bucket-list'][add_index]
    product_relative_category_price = x['mean_price_hierarchy-list'][add_index]
    product_relative_main_category_price = x['mean_price_main-list'][add_index]
    
    return (add_product_id, add_nb_interactions,
            has_been_detailed, has_been_removed, has_been_viewed, has_been_searched, has_been_clicked,
            product_category, product_main_category, product_price, product_relative_category_price, product_relative_main_category_price)

def session_feartures(x): 
    session_len = np.sum(x['product_nb_interactions-list'])
    unique_items =  len(set(x['product_url_hash_list']))
    return [ session_len, unique_items, x['nb_queries']]

def get_p_interactions(x, p=5): 
    cols = ['product_url_hash_list', 'event_type-list',
            'product_action-list', 'category_hash-list',
           'price_bucket-list', 'mean_price_hierarchy-list']
    add_product_id = x['first_AC_id-last']
    add_index = x['product_url_hash_list'].tolist().index(add_product_id)
    
    last_p_interactions_after  = [x[col][add_index+1:][-p:] for col in cols]
    first_p_interactions_before  = [x[col][:add_index][:p] for col in cols]
    return  last_p_interactions_after + first_p_interactions_before
