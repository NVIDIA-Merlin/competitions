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

import argparse
import json
import os
import random
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn.metrics as metrics
import yaml
from evaluation.evaluation import cart_abandonment_metric
from submission.uploader import upload_submission

import xgboost as xgb
from main import load_fold_data, set_seeds

  
def objective(trial, args):
    """
    Compute the CV metric to optimize
    """
    # load feature name 
    with open(args.feature_config) as f:
        X_COLS = f.read().splitlines() 
        
    # load data 
    bag_number = args.bag_number
    fold = args.fold_number
    seed = (bag_number * 10) 
    set_seeds(seed)
    data_path = args.data_path
    sigir_metric = []

    # Compute average score of the cross-validation folds 
    metric_to_optim = []
    for fold in range(1, args.num_folds+1): 
        # set CV files 
        VALID_PARQUET_FILE = os.path.join(data_path, f"valid-{fold}.parquet")
        training_datasets = ["train"]
        training_files = []
        test_files = []
        for f in range(1, args.num_folds + 1):
            if f != fold:
                for dataset in training_datasets:
                    training_files.append(os.path.join(data_path, f"{dataset}-{f}.parquet"))
        TRAINING_PARQUET_FILES = training_files
        random.shuffle(TRAINING_PARQUET_FILES)
        TEST_FULL_PARQUET_FILE = os.path.join(data_path, f"test-full.parquet")
        # load XGB data matrices 
        subgroupb_XGB_data = load_fold_data(TRAINING_PARQUET_FILES,
                                            [VALID_PARQUET_FILE],
                                            [TEST_FULL_PARQUET_FILE],
                                            X_COLS,
                                            args.subgroup_models)
        
        # Set xgb parameters for the trial 
        xgb_parms = {
                    "max_depth": trial.suggest_int('max_depth', 2, 20, 2),
                    "learning_rate": trial.suggest_loguniform('learning_rate', 0.01, 0.8),
                    "subsample": trial.suggest_discrete_uniform('subsample', 0.2, 1, 0.2),
                    "colsample_bytree": trial.suggest_discrete_uniform('colsample_bytree', 0.2, 1, 0.2) ,
                    "reg_lambda": trial.suggest_int('reg_lambda', 1, 10, 1) ,
                    "eval_metric": "auc",
                    "objective": "binary:logistic",
                    "tree_method": "gpu_hist",
                    "predictor": "gpu_predictor",
                    "verbosity": 0,
                    "seed": seed}
        
        #training & evaluating 
        print("Training...")
        results = get_cv_metrics(subgroupb_XGB_data, xgb_parms, num_round= trial.suggest_int('num_rounds', 10, 500, 10))
        
        if args.optim_metric == 'cart': 
             metric_to_optim.append(results['Weighted micro F1-score'])
        elif args.optim_metric == 'auc': 
            metric_to_optim.append(results['Weighted micro F1-score'])
    return np.mean(metric_to_optim)


def get_cv_metrics(subgroupb_XGB_data, xgb_parms, num_round):
    eval_per_group = []
    for subgroup, data in subgroupb_XGB_data.items():
        (X_train, y_train, eval_info_train), (X_validation, y_validation, validation_predictions_df), (X_test, y_test, test_predictions_df) = data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalidation = xgb.DMatrix(X_validation, label=y_validation)
        dtest = xgb.DMatrix(X_test, label=y_test) 
        #Training 
        evallist = [(dtrain, 'train'), (dvalidation, 'validation')]
        early_stopping_round = 5
        bst = xgb.train(xgb_parms, dtrain, num_round, evals=evallist,  early_stopping_rounds=early_stopping_round)
            
        # build validation prediction  
        eval_preds = bst.predict(dvalidation, iteration_range=(0, bst.best_iteration + 1))
        threshold = 0.5
        eval_preds_class = (eval_preds > threshold).reshape(-1).astype(int).tolist()
        validation_predictions_df.columns = ['session_id_hash', 'nb_after_add']
        validation_predictions_df['predicted_label'] = eval_preds_class
        validation_predictions_df['predicted_prob'] = eval_preds
        validation_predictions_df['label'] = y_validation
        eval_per_group.append(validation_predictions_df)
                
    total_validation = pd.concat(eval_per_group)
        
    # Compute cv metrics 
    fpr, tpr, thresholds = metrics.roc_curve(total_validation['label'].values, total_validation['predicted_prob'].values)
    auc = metrics.auc(fpr, tpr)
    metric_result = cart_abandonment_metric(preds=total_validation['predicted_label'].values,
                                            labels=total_validation['label'].values,
                                            nb_after_add=total_validation['nb_after_add'].values)

    results = {'Weighted micro F1-score': metric_result,  'auc': auc}
    return results


def main(args):
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective,  args=args), n_trials=args.n_trials)
    print("Best trial:")
    trial = study.best_trial
    print("  F1-Score: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--subgroup_models', action='store_true',
                        help='Enable training multiple XGB for different nb_after_add subgroups'
                        )
    
    parser.add_argument('--bag_number',  type=int,  default=1,
                        help='Number of training bags (each bag will use different seeds')

    parser.add_argument('--fold_number',  type=int, 
                        help='Number of folds for each bag')
    
    parser.add_argument('--num_folds',  type=int, 
                        help="Number of folds for each bag")

    parser.add_argument('--output_dir',  type=str, 
                        help='output folder where results are saved')
    
    parser.add_argument('--data_path',  type=str, 
                        help='path to data parquet files ')
    
    parser.add_argument('--feature_config',  type=str, 
                        help='path to text file specifying the name of columns to use')
    
    parser.add_argument('--n_trials', type=int, default=100, 
                        help='number of bayesian round for hyperparam tuning'
                        )
    parser.add_argument('--optim_metric', type=str, default='cart', 
                       help='name of the metric to optimize : ["cart", "auc"]')

    args = parser.parse_args()
    
    main(args)