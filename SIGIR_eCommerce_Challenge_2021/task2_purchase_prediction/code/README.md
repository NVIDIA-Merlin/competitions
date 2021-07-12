# SIGIR eCommerce Data Challenge 2021 - Task (2)  Purchase Intent Prediction
This subdirectory contains the code related to XGB and dlrm solutions used for the Task (2) Purchase Intent Prediction of the SIGIR eCommerce Data Challenge 2021.

# Setup with Conda

1- Create a Conda environment

2- [Install NVTabular](https://github.com/NVIDIA/NVTabular/#installing-nvtabular-using-conda) using conda like this (TODO: Fix NVTabular version to 0.6 as soon as it is released in July).
```bash
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=11.2
```

3 - Install the project requirements
```bash
pip install -r code/requirements.txt 
```

# Preprocessing and feature engineering

For both preprocessing versions: with (V1) and without repetitions (V2), The ETL pipeline is available in Jupyter Notebooks and is composed by two steps:
  - [coveo-ETL-NVT-Task2-V*-phase2.ipynb](preprocessing/preproc-V2/coveo-ETL-NVT-Task2-V2-phase2.ipynb): Generates dataframe with session-level features.
  - [XGB_datasets.ipynb](preprocessing/preproc-V2/XGB_datasets.ipynb): Generates tabular features for different dataset profiles based on nb_after_add distribution. The list of generated features is detailed in Table 2 of the [README](https://github.com/rapidsai/recsys/tree/main/competitions/sigir_ecom_challenge_2021_for_release/task2_purchase_prediction)
  
# Training, evaluation and prediction generation of XGB model

- The main script: `xgboost/main.py` handles all steps of the pipeline:  Training, Evaluation, Feature Improtance, Feature Selection and Prediction generation.

- The training of one general xgb model or a model per nb_after_add subgroup is controlled by the bool argument `--subgroup_models`

- Feature selection using random swapping noise is enabled using the bool argument --swap_noise_selection

- The following command is an example of how to train an XGB model, generate predictions parquet file and select important features:

```bash
python main.py --bag_number 1 --output_dir result_dir/ --data_path /data/sessions_wo_repetitions/xgboost_data/without_truncation --compute_feature_importance  --swap_noise_selection --num_folds 5 --num_round 20  --max_depth 6 --learning_rate 0.5295173477082308 --colsample_bytree 0.4 --reg_lambda 6 --subsample 0.8 --feature_config $PATH/features/cart_features_98.txt
```  

- Documentation about remaining arguments (prefixed by --) can be found in [main.py](xgboost/main.py). 

- Similar to Task 1, to generate a submission file in the format required by the competition (JSON) with predictions for test set sessions, set the `--do_submit` bool argument and the `--email_submission` argument with the e-mail used to register for the competition.

# XGB hyperparameter tuning 
- Hypertuning of XGB model is conducted using the bayesian optimization package `optuna` and the script to launch the hypertuning experiment can be found at [hypertuning_xgboost.py](xgboost/hypertuning_xgboost.py), here is an example of the command line: 
```bash
python hypertuning_xgboost.py --bag_number 1 --output_dir tmp --data_path /data/sessions_wo_repetitions/xgboost_data/duplicated_sessions_with_truncation --num_folds 5  --n_trials 100  --feature_config cart_xgboost_features-TE_prod_id.txt
```
- The hyperparamters search space and results are detailed in section [5.1. Hyperparameter Tuning](https://github.com/rapidsai/recsys/tree/main/competitions/sigir_ecom_challenge_2021_for_release/task2_purchase_prediction)

# Training, evaluation and prediction generation of DLRM model

This DLRM model is built and trained with `Keras` library using `tf.keras` API on GPU. Instead of native Tensorflow dataloaders, we used optimized [NVTabular Tensorflow dataloader](https://github.com/NVIDIA/NVTabular/blob/main/nvtabular/loader/tensorflow.py). In order to train the model, Tensorflow 2 library should be installed. You can follow the instructions above in the `Setup with Conda` Section to install Tensorflow 2 in a conda environment. 

If you would like to install NVTabular and Tensorflow via `docker`, you can use [merlin-tensorflow-training](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training) docker image available in [NVIDIA Merlin container repository](https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training). 


- The main script: `dlrm/tfdlrm-sigir.py` handles the following steps of the pipeline:  Training, Evaluation and Prediction generation.
- `--data_path` argument defines the path of the processed parquet files. See [README](https://github.com/rapidsai/recsys/blob/main/competitions/sigir_ecom_challenge_2021_for_release/task2_purchase_prediction/README.md) for the details about the preprocessing and feature engineering steps.
- Numerical and categorical features should be fed in different files. `--feature_config_cat` argument is used to feed categorical features as a text file, whereas `--feature_config_num` is used for numerical features text file.
- `--output_dir` is set to define the path where to save the generated prediction files for each `fold` and `bag` to disk. 
- `--save_pred` argument is set to save the generated parquet files for each `fold` and each `bag_number` to disk. If not used, parquet files will not be saved.
- The following command is an example of how to train an DLRM model, generate predictions parquet file and save those files to disk:

```
python tfdlrm-sigir.py --output_dir tmp_dlrm --data_path /workspace/SIGIR-ecom-data-challenge/data/ --lr 0.001 --feature_config_cat cat_features_dlrm.txt --feature_config_num num_features_dlrm.txt --emb_dim 64 --fold 1 --bag_number 1 --epochs 3 --save_pred
```

## Submitting the prediction results to Leader Board

In order to submit the prediction results, generated from DLRM model, to leader board we used `dlrm/submit.py`. 

- `--data_path` is the path to the bag directory generated by `tflrm-sigir.py` and set to read the saved parquet files for each fold under the given `bag_number` folder. Note that each bag directory includes a folder for each fold. Once the prediction parquet file of each fold is read, the mean of the predictions is calculated as a final ensemble prediction file, and saved to ` --output_dir` under the given `bag_number` folder. 
- `--test_path` argument defines the path of the test data `intention_test_phase_2.json` file.

The following command is an example of how to submit the predictions to leader board:

```
python submit.py --data_path /workspace/SIGIR-ecom-data-challenge/recsys/competitions/sigir_ecom_challenge_2021/xgboost/tmp_dlrm/bag_1/ --test_path /workspace/SIGIR-ecom-data-challenge/coveo_task2_v1_phase2/xgboost_data/ --output_dir tmp_dlrm  --bag_number 1 --email_submission <user_email> --do_submit
```
