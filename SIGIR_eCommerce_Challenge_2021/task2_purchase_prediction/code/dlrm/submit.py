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

'''
the example script to run this file:

python submit.py --data_path /workspace/SIGIR-ecom-data-challenge/data/tmp_dlrm/bag_4/ --test_path /workspace/SIGIR-ecom-data-challenge/data/ --output_dir tmp_dlrm  --bag_number 4 --email_submission <user_email> --do_submit
'''

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from evaluation import cart_abandonment_metric
from uploader import upload_submission

import xgboost as xgb


def load_test_data(data_path=None): 
    fold_folders = [f for f in sorted(os.listdir(data_path))]
    cv_preds = []
    for fold_folder in fold_folders:
        test_predictions_df = pd.read_parquet(os.path.join(data_path, fold_folder, 'test_predictions_dlrm.parquet'))
        cv_preds.append(test_predictions_df['predictions'].values)
    # ensemble predictions of the five folds
    test_predictions_df['predictions'] = np.mean(np.array(cv_preds), axis=0)
    test_predictions_df.rename(columns={"original_session_id_hash": "session_id_hash"}, inplace=True)
    return test_predictions_df, cv_preds


def generate_submission_file(test_predictions_df, threshold, data_path): 
    #load json file 
    with open(os.path.join(data_path, "intention_test_phase_2.json")) as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)
        test_df = pd.json_normalize(test_queries, 'query', 'nb_after_add')
        test_df = test_df.drop_duplicates('session_id_hash')
        
    assert len(test_predictions_df) == len(test_df)
    #merge predictions frame and provided test_df to insure same order of sessions 
    test_df = test_df.merge(test_predictions_df, on='session_id_hash', how='left')
    
    
    preds = (test_df.predictions.values > threshold).reshape(-1).astype(int).tolist()
    print("Number of purchases predicted in test set is: %s" %np.sum(preds))
    
    # Convert to required prediction format
    preds = [{'label':pred} for pred in preds]
    
    
    local_prediction_file = "{}_{}.json".format(
        args.email_submission.replace("@", "_"), round(time.time() * 1000)
    )
    
    local_prediction_file_path = os.path.join(
        args.output_dir, f"bag_{args.bag_number}", local_prediction_file
    )
    print("Generating JSON file with predictions")
    with open(local_prediction_file_path, "w") as fp:
        json.dump(preds, fp, indent=2)

    return local_prediction_file_path

def main(args):

    ################# SIGIR eCom Data Challenge - k-fold train #################


    bag_number = args.bag_number

    output_bag_folder = os.path.join(args.output_dir, f"bag_{bag_number}")
    print("output_bag_folder", output_bag_folder)

    data_path = args.data_path
    
    sigir_metrics, aucs, cv_preds = [], [], []


    # load data 

    test_predictions_df, cv_preds = load_test_data(data_path=data_path)

    # Generate submission file 
    # ensemble predictions of the five folds
    local_prediction_file_path = generate_submission_file(test_predictions_df, 0.5, args.test_path)
    test_predictions_df.to_parquet(os.path.join(output_bag_folder, "ensemble_predictions.parquet"))
    print("Ensemble cv predictions saved to: %s" %local_prediction_file_path)
    
    if args.do_submit:
        print(f"************* Uploading the submission file *************")
        TASK = "cart"  # 'rec' or 'cart'
        upload_submission(local_file=local_prediction_file_path, task=TASK)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bag_number',  type=int,  default=1,
                        help='Number of training bags (each bag will use different seeds')

    parser.add_argument('--num_folds',  type=int, default=5,
                        help="Number of folds for each bag")

    parser.add_argument('--output_dir',  type=str, 
                        help='output folder where results are saved')
    
    parser.add_argument('--data_path',  type=str, 
                        help='path to data parquet files')
    
    parser.add_argument('--test_path',  type=str, 
                        help='path to test json file')
    
    parser.add_argument('--feature_config',  type=str, 
                        help='path to text file specifying the name of columns to use')


    # xgboost params
    parser.add_argument('--max_depth',  type=int, default=3,
                        help='Maximum depth of a tree.')
    
    parser.add_argument('--num_round',  type=int,  default=10,
                        help='The number of rounds for boosting')
    
    parser.add_argument('--subsample',  type=float,  default=1,
                        help='Subsample ratio of the training instances.')
    
    parser.add_argument('--colsample_bytree', type=float,  default=1,
                       help='The subsample ratio of columns when constructing each tree')
    
    parser.add_argument('--learning_rate',  type=float, default=0.5, 
                        help='Step size shrinkage used in update to prevents overfitting.')
        
    parser.add_argument('--reg_lambda', type=float, default=1,
                        help='L2 regularization term on weights.'
                        )
    
    parser.add_argument('--scale_pos_weight', type=int, default=1, 
                        help='Control the balance of positive and negative weights, useful for unbalanced classes. '
                        )

    # submission params
    parser.add_argument('--email_submission',  type=str, 
                        )
    parser.add_argument('--do_submit', action='store_true',
                        )

    parser.add_argument('--train_full', action='store_true',
                        )
    args = parser.parse_args()
    
    main(args)
