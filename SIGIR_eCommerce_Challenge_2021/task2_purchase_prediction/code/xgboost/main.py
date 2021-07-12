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

import matplotlib.pyplot as plt
import numpy as np
import nvtabular as nvt
import pandas as pd
import sklearn.metrics as metrics
import yaml
from evaluation.evaluation import cart_abandonment_metric
from submission.uploader import upload_submission

import xgboost as xgb


def load_fold_data(train_files, valid_files, test_files ,X_COLS, subgroup_models=False, 
                   TE_COLS= ['add_product_id', 
                            'product_url_id_list_after-0',
                            'product_url_id_list_after-1',
                            'product_url_id_list_after-2',
                            'product_url_id_list_after-3',
                            'product_url_id_list_after-4', 
                            'product_url_id_list_before-0',
                            'product_url_id_list_before-1',
                            'product_url_id_list_before-2',
                            'product_url_id_list_before-3',
                            'product_url_id_list_before-4']): 
    
    train_frames = pd.concat([pd.read_parquet(file) for file in train_files])
    valid_frames = pd.concat([pd.read_parquet(file) for file in valid_files])
    test_frames = pd.concat([pd.read_parquet(file) for file in test_files])
    if len(TE_COLS) > 0:
        print("Target encoding of columns {}\n".format(TE_COLS))
        # target encoding 
        other_cols = [col for col in train_frames.columns if col not in TE_COLS] 
        te_features = TE_COLS >> nvt.ops.TargetEncoding("is_purchased-last", p_smooth=20)
        df_train = nvt.Dataset(train_frames)
        workflow = nvt.Workflow(other_cols + te_features)
        # fit TE model
        workflow.fit(df_train)
        # transform features 
        train_frames = workflow.transform(nvt.Dataset(train_frames)).to_ddf().compute().to_pandas()
        valid_frames = workflow.transform(nvt.Dataset(valid_frames)).to_ddf().compute().to_pandas()
        test_frames = workflow.transform(nvt.Dataset(test_frames)).to_ddf().compute().to_pandas()
    
    subgroupb_XGB_data = {}
    if subgroup_models: 
        for subgroup in [0, 2, 4, 6, 8, 10]: 
                # return arrays for train / eval 
                y_train = train_frames[train_frames['nb_after_add-last']==subgroup]['is_purchased-last'].values 
                y_valid = valid_frames[valid_frames['nb_after_add-last']==subgroup]['is_purchased-last'].values 
                y_test = test_frames[test_frames['nb_after_add-last']==subgroup]['is_purchased-last'].values 
                X_train = train_frames[train_frames['nb_after_add-last']==subgroup][X_COLS].values
                X_valid = valid_frames[valid_frames['nb_after_add-last']==subgroup][X_COLS].values
                X_test = test_frames[test_frames['nb_after_add-last']==subgroup][X_COLS].values
                # session id info 
                eval_information = valid_frames[valid_frames['nb_after_add-last']==subgroup][['original_session_id_hash',  'nb_after_add-last']]
                test_information = test_frames[test_frames['nb_after_add-last']==subgroup][['original_session_id_hash',  'nb_after_add-last']]
                train_information = train_frames[train_frames['nb_after_add-last']==subgroup][['original_session_id_hash',  'nb_after_add-last']]
                subgroupb_XGB_data[subgroup] = (X_train, y_train, train_information), (X_valid, y_valid, eval_information), (X_test, y_test, test_information)
    
    else: 
        # return arrays for train / eval 
        y_train = train_frames['is_purchased-last'].values 
        y_valid = valid_frames['is_purchased-last'].values 
        y_test = test_frames['is_purchased-last'].values 
        X_train = train_frames[X_COLS].values
        X_valid = valid_frames[X_COLS].values
        X_test = test_frames[X_COLS].values
        # session id info 
        eval_information = valid_frames[['original_session_id_hash',  'nb_after_add-last']]
        test_information = test_frames[['original_session_id_hash',  'nb_after_add-last']]
        train_information = train_frames[['original_session_id_hash',  'nb_after_add-last']]
        subgroupb_XGB_data['all'] = (X_train, y_train, train_information), (X_valid, y_valid, eval_information), (X_test, y_test, test_information)

    return subgroupb_XGB_data


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed+1)


def main(args):
    """
       SIGIR eCom Data Challenge - k-fold train and test module 
       
    """
    # load feature name to use for fitting XGB model from txt file 
    with open(args.feature_config) as f:
        X_COLS = f.read().splitlines() 
        print("\nUse XGboost with features:\n {}\n".format(X_COLS))
    
    # Fix the experiment seed 
    bag_number = args.bag_number
    seed = (bag_number * 10) 
    set_seeds(seed)
    
    # Init output metrics 
    output_bag_folder = os.path.join(args.output_dir, f"bag_{bag_number}")
    sigir_metrics, aucs, cv_preds = [], [], []
    
    data_path = args.data_path
    
    for fold in range(1, args.num_folds+1):
        print(f"Starting BAG {bag_number} - FOLD {fold}")
        output_fold_folder = os.path.join(output_bag_folder, f"fold_{fold}")
        os.makedirs(output_fold_folder, exist_ok=True)
        # Load XGB datasets
        print(f"Loading DATA from {data_path}\n")
        data_path = args.data_path
        VALID_PARQUET_FILE = os.path.join(data_path, f"valid-{fold}.parquet")
        TEST_FULL_PARQUET_FILE = os.path.join(data_path, f"test-full.parquet")
        training_datasets = ["train"]
        training_files = []
        for f in range(1, args.num_folds + 1):
            if f != fold:
                for dataset in training_datasets:
                    training_files.append(os.path.join(data_path, f"{dataset}-{f}.parquet"))
        TRAINING_PARQUET_FILES = training_files
        random.shuffle(TRAINING_PARQUET_FILES)

        print(f"\t\tTrain parquet files: {TRAINING_PARQUET_FILES}\n")
        print(f"\t\tValid parquet file: {VALID_PARQUET_FILE}\n")
        print(f"\t\tTest parquet files: {TEST_FULL_PARQUET_FILE}\n")

        # Train XGB models 
        eval_per_group = []
        test_per_group = []
        subgroupb_XGB_data = load_fold_data(TRAINING_PARQUET_FILES, [VALID_PARQUET_FILE], [TEST_FULL_PARQUET_FILE] ,X_COLS, args.subgroup_models)
        for subgroup, data in subgroupb_XGB_data.items():
            (X_train, y_train, eval_info_train), (X_validation, y_validation, validation_predictions_df), (X_test, y_test, test_predictions_df) = data
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalidation = xgb.DMatrix(X_validation, label=y_validation)
            dtest = xgb.DMatrix(X_test, label=y_test)
            # Define xgboost model 
            xgb_parms = {
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "reg_lambda": args.reg_lambda,
                "scale_pos_weight": args.scale_pos_weight,
                "eval_metric": "auc",
                "objective": "binary:logistic",
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "seed": seed}
            
            #Training 
            print("Training for subgroup: %s "%subgroup)
            evallist = [(dtrain, 'train'), (dvalidation, 'validation')]
            early_stopping_round = 5
            bst = xgb.train(xgb_parms, dtrain, args.num_round, evals=evallist,  early_stopping_rounds=early_stopping_round)
            
            # build validation prediction  
            eval_preds = bst.predict(dvalidation, iteration_range=(0, bst.best_iteration + 1))
            threshold = 0.5
            eval_preds_class = (eval_preds > threshold).reshape(-1).astype(int).tolist()
            validation_predictions_df.columns = ['session_id_hash', 'nb_after_add']
            validation_predictions_df['predicted_label'] = eval_preds_class
            validation_predictions_df['predicted_prob'] = eval_preds
            validation_predictions_df['label'] = y_validation
            eval_per_group.append(validation_predictions_df)
            
            #Predicting 
            test_preds = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
            # Combining predictions with session hash ids
            test_predictions_df.columns = ['session_id_hash', 'nb_after_add']
            test_predictions_df["predicted_prob"] = list(test_preds)
            test_per_group.append(test_predictions_df)
            
        total_validation = pd.concat(eval_per_group)
        test_predictions_df = pd.concat(test_per_group)
        
        # Compute cv metrics 
        print("Compute evaluation metrics using fold %s..."%fold)
        fpr, tpr, thresholds = metrics.roc_curve(total_validation['label'].values, total_validation['predicted_prob'].values)
        auc = metrics.auc(fpr, tpr)
        metric_result = cart_abandonment_metric(preds=total_validation['predicted_label'].values,
                                                labels=total_validation['label'].values,
                                                nb_after_add=total_validation['nb_after_add'].values)
        results = {'Weighted micro F1-score': metric_result,  'auc': auc}
        print("\t\tEvaluation metrics: {}".format(results))
        sigir_metrics.append(metric_result)
        aucs.append(auc)
        
        
        # Save validation predictions to parquet 
        total_validation.to_parquet(os.path.join(output_fold_folder, "valid_predictions.parquet"))
        cv_preds.append(test_predictions_df["predicted_prob"].values)
        print("\n")

    results = {'Weighted micro F1-score': np.mean(sigir_metrics),  'auc': np.mean(aucs)}
    print('Cross-validated metrics: {}\n'.format(results))

    # Get Feature importance 
    if args.compute_feature_importance: 
        print(f"************* Get feature importance *************")
        feature_importance = bst.get_score(importance_type='gain')
        keys = list(feature_importance.keys())
        values = list(feature_importance.values())
        keys = np.array(X_COLS)[[int(x.split('f')[-1]) for x in keys]]
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
        print(data)
        data.to_csv(os.path.join(output_fold_folder, "feature_importance.csv"))

    if args.swap_noise_selection:
        print(f"************* Feature selection using swap noise *************")
        swap_noise_feature_selection(X_COLS, X_validation, y_validation,
                                     validation_predictions_df['nb_after_add'].values,
                                     bst, threshold, output_bag_folder)
    # Generate submission file     
    if args.do_submit:
        # Ensemble predictions from the five folds
        test_predictions_df['predictions'] = np.mean(np.array(cv_preds), axis=0)
        local_prediction_file_path = generate_submission_file(test_predictions_df, threshold, args.data_path )
        test_predictions_df.to_parquet(os.path.join(output_bag_folder, "ensemble_predictions.parquet"))
        print("Ensemble cv predictions saved to: %s" %local_prediction_file_path)
        print(f"************* Uploading the submission file *************")
        TASK = "cart"  # 'rec' or 'cart'
        upload_submission(local_file=local_prediction_file_path, task=TASK)

def swap_noise_feature_selection(X_COLS, X_validation, y_validation, nb_after_add, bst, threshold, output_bag_folder):
    # Compute cv metrics by swaping columns 
    features_scores = {}
    for i in range(len(X_COLS)): 
        print("shuffling the column: %s"%X_COLS[i])
        validation = np.copy(X_validation)
        validation[:, i] = random.shuffle(validation[:, i])
        dvalidation = xgb.DMatrix(validation, label=y_validation)
        eval_preds = bst.predict(dvalidation, iteration_range=(0, bst.best_iteration + 1))
        fpr, tpr, thresholds = metrics.roc_curve(y_validation, eval_preds)
        auc = metrics.auc(fpr, tpr)
        eval_preds_class = (eval_preds > threshold).reshape(-1).astype(int).tolist()
        metric_result = cart_abandonment_metric(preds=eval_preds_class, labels=y_validation,
                                    nb_after_add=nb_after_add)
        results = {'Weighted micro F1-score': metric_result,  'auc': auc}
        print("\t\tEvaluation metrics: {}".format(results))
        features_scores[X_COLS[i]] = results
    with open(os.path.join(output_bag_folder, "feature_selection_swaping_noise_score.json"), "w") as fp:
        json.dump(features_scores, fp, indent=2)
    print("Validation scores with swap noise saved at : %s" %os.path.join(output_bag_folder, "feature_selection_swaping_noise_score.json"))
        
        
def get_best_thershold(thresholds, preds, labels, after_adds):
    """
    Get best threshold optimizing the cart_abandonment_metric
    """
    cart_metrics = []
    for tr in thresholds:
        cart_metrics.append(cart_abandonment_metric(preds=(preds > tr).reshape(-1).astype(int).tolist(),
                                                labels=labels,
                                                nb_after_add=after_adds))
    ix = np.argmax(cart_metrics)
    return thresholds[ix]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # data params 
    parser.add_argument('--bag_number',  type=int,  default=1,
                        help='Number of training bags (each bag will use different seeds')

    parser.add_argument('--num_folds',  type=int, default=5,
                        help="Number of folds for each bag")

    parser.add_argument('--output_dir',  type=str, 
                        help='output folder where results are saved')
    
    parser.add_argument('--data_path',  type=str, 
                        help='path to data parquet files ')
    
    parser.add_argument('--feature_config',  type=str, 
                        help='path to text file specifying the name of columns to use')

    # xgboost params
    parser.add_argument('--subgroup_models', action='store_true',
                        help='Enable training multiple XGB for different nb_after_add subgroups'
                        )
    
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
        
    parser.add_argument('--compute_feature_importance', action='store_true',
                        help='Enable computing feature importance from fitted xgb model'
                        )
          
    parser.add_argument('--swap_noise_selection', action='store_true',
                        help='Enable feature selection using swapping noise technique'
                        )

    # submission params
    parser.add_argument('--email_submission',  type=str,
                        help= 'login email for AWS submission'
                        )
    parser.add_argument('--do_submit', action='store_true',
                        help = "Enable AWS submission to get LB score"
                        )
    
    args = parser.parse_args()
    
    main(args)

