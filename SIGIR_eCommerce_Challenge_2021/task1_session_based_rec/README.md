# SIGIR eCommerce Data Challenge 2021 - Task (1) Session-based Recommendation

This folder contains NVIDIA Merlin team solution for the Task (1) Session-based recommendation of the SIGIR eCommerce Data Challenge 2021, which placed 1st in the **Subsequent Items Prediction** leaderboard and 2nd in the **Next Item Prediction**  leaderboard. 

In this task models are evaluated for their ability to predict the *immediate next product* interacted by the user in a session *Mean Reciprocal Rank - MRR* and to predict *all subsequent interacted products* in the session, up to a maximum of 20 after the current event (*F1 score*). 

This repo includes Python code, scripts and instructions on how to run the pre-processing, training, evaluation and prediction pipeline. The solution is described in this our [paper](https://sigir-ecom.github.io/) for SIGIR eCom'21 workshop, including the details of the model architectures, shown in Figure 1.

<p align="center">
  <img src="https://github.com/NVIDIA-Merlin/competitions/blob/main/SIGIR_eCommerce_Challenge_2021/task1_session_based_rec/images/sigir_ecom_architecture.PNG" width="600" />
  <br>
  <font size="1">Fig. 1 - Our base Transformer architecture</font>
</p>

## Setup the environment

The next sections describe two alternatives to setup the environment to run the pipelines: Docker and Conda.

### Setup with Docker
This section contains instructions on how to build and start a Docker container to run the training, evaluation and prediction pipelines. It requires Docker and 
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

#### Build the container
```bash
cd task1_session_based_rec/
docker build --no-cache --tag sigir_ecom_dev -f container/Dockerfile .
```

#### Run the container
The following command shows how to run the Docker container you have build in the previous step. The source code will be mounted within the container in the `\sigir_ecom` mount and the preproc data in `\data` mount. The port 8888 is also mapped so that Jupyter notebook can be run within the container to execute the preprocessing notebooks.

```bash
cd task1_session_based_rec/
DATA_PATH=~/dataset/ #Path to the folder container the competition data
docker run --gpus all -it --rm -p 8888:8888 -v ${DATA_PATH}:/data -v $(pwd):/sigir_ecom --workdir /sigir_ecom/ sigir_ecom_dev /bin/bash
```

The commands to run the pipelines are explained in the next sections.

### Setup with Conda

1 - Create a Conda environment  
2 - [Install NVTabular](https://github.com/NVIDIA/NVTabular/#installing-nvtabular-using-conda) using conda like this (TODO: Fix NVTabular version to 0.6 as soon as it is released in July).
```bash
conda install -c nvidia -c rapidsai -c numba -c conda-forge nvtabular python=3.7 cudatoolkit=11.2
```
3 - Install the project requirements
```bash
pip install -r sigir_ecom_transformers/requirements.txt 
```

## Preprocessing and feature engineering

The ETL pipeline is available in Jupyter Notebooks and is composed by two steps:

1 - [ecom_preproc_step1.ipynb](1-preprocessing/ecom_preproc_step1.ipynb) - First step of the preprocessing  
2 - [ecom_preproc_step2.ipynb](1-preprocessing/ecom_preproc_step2.ipynb) and [ecom_preproc_step2-item_freq_capping.ipynb](1-preprocessing/ecom_preproc_step2-item_freq_capping.ipynb) - Refines the preprocessing, including the split of validation sessions into two halves (the first half for inference and second half for metrics evaluation). The difference between the Step 2 notebooks is that the `ecom_preproc_step2-item_freq_capping.ipynb` (used only by the XLNET-IM-FC architecture) applies item frequency capping, replacing item ids with have less than 5 interactions by an item id reserved for unfrequent items. 

The preprocessing notebook uses RAPIDS (cudf, cupy) and NVTabular, so need to be run either within the Docker image or in a conda envirnoment with such frameworks installed.


## Training, evaluation and prediction generation

The training pipeline (in [2-models](2-models/) folder) can be run under two configurations:
- **Evaluation pipeline** - Pre-trains the model with train sessions and fine-tunes with validation sessions (first half / session beginning). The second half of sessions (session end) is used to compute the metrics (MRR and F1). In the output folder, metric results are saved to a file named `eval_train_results.csv` and predictions for the fold validation set are saved to `valid_eval_predictions.parquet`.
- **Test prediction pipeline** - Pre-trains the model with train and validation sessions and fine-tunes with test sessions. In the output folder, predictions for the test set are saved to a file named `test_predictions.parquet`.

The following command is an example to run the pipelines. For the evaluation pipeline set  variable `SCRIPT_NAME` to `run_sigir_pipeline_eval.bash` and for test predictions pipeline set to `run_sigir_pipeline_test_preds.bash`.

Both bash scripts have 3 required parameters:
- `CUDA_VISIBLE_DEVICES_GPUS` - Id of the GPU device that should be used for the run. This pipeline was only tested with a single GPU (V100 32 GB RAM)
- `FOLD_NUMBER` - Int number between 1-5, used to train the model with Out-Of-Fold (OOF) data.
- `DATA_PATH` - Path of preprocessed data (in Parquet format)

```bash
cd 2-models/

SCRIPT_NAME=run_sigir_pipeline_eval.bash #or run_sigir_pipeline_test_preds.bash
CUDA_VISIBLE_DEVICES_GPUS=0
FOLD_NUMBER=1
DATA_PATH=/data
bash scripts/${SCRIPT_NAME} ${CUDA_VISIBLE_DEVICES_GPUS} ${FOLD_NUMBER} ${DATA_PATH} --feature_config datasets/sigir_ecommerce/config/features/session_based_features_all.yaml --fp16 --model_type xlnet --loss_type cross_entropy --per_device_eval_batch_size 512 --similarity_type concat_mlp --tf_out_activation tanh  --inp_merge mlp --learning_rate_warmup_steps 0 --learning_rate_schedule linear_with_warmup --hidden_act gelu --num_train_epochs 3 --compute_metrics_each_n_steps 1 --mf_constrained_embeddings --layer_norm_featurewise --layer_norm_all_features --input_features_aggregation concat --num_folds 5 --bag_number 1 --fold_number 1 --predict_top_k 100 --evaluation_strategy steps --eval_steps 200 --max_eval_steps 20 --modules_merge elementwise --include_prod_description_emb_feature --include_prod_image_emb_feature --mlm --attn_type bi --per_device_train_batch_size 448 --learning_rate 0.001960283514353802 --dropout 0.0 --input_dropout 0.30000000000000004 --weight_decay 4.846329617539937e-06 --d_model 64 --item_embedding_dim 320 --n_layer 2 --n_head 2 --label_smoothing 0.1 --stochastic_shared_embeddings_replacement_prob 0.0 --item_id_embeddings_init_std 0.06999999999999999 --other_embeddings_init_std 0.02 --mlm_probability 0.30000000000000004 --embedding_dim_from_cardinality_multiplier 5.0 --finetuning_freeze_all_layers_by_item_id_embedding --num_epochs_finetuning 10 --learning_rate_finetuning 0.0001960283514353802
```

Documentation about remaining arguments (prefixed by --) can be found in [recsys_args.py](2-models/sigir_ecom_transformers/recsys_args.py).  
In particular, to generate a submission file in the format required by the competition (JSON) with predictions for test set sessions, set the `--do_submit` bool argument and the `--email_submission` argument with the e-mail used to register for the competition.
The arguments (hyperparameters) for each model used in the ensemble are in the following section.

## Hyperparameters of the architectures used in the Ensemble
In the following table, we present the hyperparamters of the 12 architectures used in the final ensemble. For each architecture we train one model for each fold.

<table>
<thead>
  <tr>
    <th>Model name</th>
    <th>XLNET-IM-1</th>
    <th>XLNET-IM-2</th>
    <th>XLNET-IM-3</th>
    <th>XLNET-S-1</th>
    <th>XLNET-S-2</th>
    <th>XLNET-S-3</th>
    <th>XLNET-IM-FC-1</th>
    <th>XLNET-IM-FC-2</th>
    <th>XLNET-IM-FC-3</th>
    <th>TransfoXL-IM-1</th>
    <th>TransfoXL-IM-2</th>
    <th>TransfoXL-IM-3</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>--model_type</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>xlnet</td>
    <td>transfoxl</td>
    <td>transfoxl</td>
    <td>transfoxl</td>
  </tr>
  <tr>
    <td>--mlm</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
  </tr>
  <tr>
    <td>--mlm_probability</td>
    <td>0.3</td>
    <td>0.3</td>
    <td>0.6</td>
    <td>0.3</td>
    <td>0.3</td>
    <td>0.7</td>
    <td>0.3</td>
    <td>0.6</td>
    <td>0.3</td>
    <td>0.15</td>
    <td>0.15</td>
    <td>0.15</td>
  </tr>
  <tr>
    <td>--n_head</td>
    <td>8</td>
    <td>2</td>
    <td>8</td>
    <td>8</td>
    <td>8</td>
    <td>2</td>
    <td>8</td>
    <td>16</td>
    <td>8</td>
    <td>4</td>
    <td>1</td>
    <td>2</td>
  </tr>
  <tr>
    <td>--n_layer</td>
    <td>2</td>
    <td>2</td>
    <td>3</td>
    <td>2</td>
    <td>2</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
    <td>3</td>
    <td>2</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>--d_model</td>
    <td>320</td>
    <td>64</td>
    <td>128</td>
    <td>320</td>
    <td>384</td>
    <td>384</td>
    <td>320</td>
    <td>320</td>
    <td>128</td>
    <td>320</td>
    <td>320</td>
    <td>320</td>
  </tr>
  <tr>
    <td>--embedding_dim_from_cardinality_multiplier</td>
    <td>1</td>
    <td>5</td>
    <td>8</td>
    <td>1</td>
    <td>10</td>
    <td>10</td>
    <td>1</td>
    <td>5</td>
    <td>5</td>
    <td>1</td>
    <td>3</td>
    <td>9</td>
  </tr>
  <tr>
    <td>--include_prod_description_emb_feature</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
  </tr>
  <tr>
    <td>--include_prod_image_emb_feature</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
  </tr>
  <tr>
    <td>--include_search_features</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>TRUE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
    <td>FALSE</td>
  </tr>
  <tr>
    <td>--input_dropout</td>
    <td>0.3</td>
    <td>0.3</td>
    <td>0.2</td>
    <td>0.3</td>
    <td>0.2</td>
    <td>0.1</td>
    <td>0.3</td>
    <td>0</td>
    <td>0.5</td>
    <td>0</td>
    <td>0.2</td>
    <td>0</td>
  </tr>
  <tr>
    <td>--item_embedding_dim</td>
    <td>448</td>
    <td>320</td>
    <td>320</td>
    <td>448</td>
    <td>384</td>
    <td>448</td>
    <td>448</td>
    <td>320</td>
    <td>384</td>
    <td>384</td>
    <td>320</td>
    <td>384</td>
  </tr>
  <tr>
    <td>--item_id_embeddings_init_std</td>
    <td>0.09</td>
    <td>0.07</td>
    <td>0.13</td>
    <td>0.09</td>
    <td>0.13</td>
    <td>0.15</td>
    <td>0.09</td>
    <td>0.13</td>
    <td>0.03</td>
    <td>0.09</td>
    <td>0.05</td>
    <td>0.05</td>
  </tr>
  <tr>
    <td>--label_smoothing</td>
    <td>0.6</td>
    <td>0.1</td>
    <td>0.2</td>
    <td>0.6</td>
    <td>0.7</td>
    <td>0.4</td>
    <td>0.6</td>
    <td>0.1</td>
    <td>0.2</td>
    <td>0.6</td>
    <td>0.3</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td>--weight_decay</td>
    <td>0.000005863</td>
    <td>0.000004846</td>
    <td>0.000001037</td>
    <td>0.000005863</td>
    <td>0.0000287</td>
    <td>0.000009248</td>
    <td>0.000005863</td>
    <td>0.00002524</td>
    <td>0.000006235</td>
    <td>0.000003963</td>
    <td>0.00007848</td>
    <td>0.00003346</td>
  </tr>
  <tr>
    <td>--learning_rate</td>
    <td>0.0005427</td>
    <td>0.00196</td>
    <td>0.001805</td>
    <td>0.0005427</td>
    <td>0.0008965</td>
    <td>0.0007346</td>
    <td>0.0005427</td>
    <td>0.001033</td>
    <td>0.0009129</td>
    <td>0.0005964</td>
    <td>0.001976</td>
    <td>0.00185</td>
  </tr>
  <tr>
    <td>--learning_rate_finetuning</td>
    <td>0.0002</td>
    <td>0.000196</td>
    <td>0.0001805</td>
    <td>0.0002</td>
    <td>0.00008965</td>
    <td>0.00007346</td>
    <td>0.0002</td>
    <td>0.0001033</td>
    <td>0.00009129</td>
    <td>0.0002</td>
    <td>0.000247</td>
    <td>0.00023125</td>
  </tr>
  <tr>
    <td>--learning_rate_schedule</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
    <td>linear_with_warmup</td>
  </tr>
  <tr>
    <td>--num_train_epochs</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
    <td>3</td>
  </tr>
  <tr>
    <td>--num_epochs_finetuning</td>
    <td>1</td>
    <td>10</td>
    <td>10</td>
    <td>1</td>
    <td>10</td>
    <td>10</td>
    <td>1</td>
    <td>10</td>
    <td>10</td>
    <td>1</td>
    <td>8</td>
    <td>8</td>
  </tr>
  <tr>
    <td>--per_device_train_batch_size</td>
    <td>384</td>
    <td>448</td>
    <td>256</td>
    <td>384</td>
    <td>448</td>
    <td>448</td>
    <td>384</td>
    <td>512</td>
    <td>384</td>
    <td>512</td>
    <td>384</td>
    <td>320</td>
  </tr>
  <tr>
    <td>--per_device_eval_batch_size</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
    <td>512</td>
  </tr>
  <tr>
    <td>--stochastic_shared_embeddings_replacement_prob</td>
    <td>0</td>
    <td>0</td>
    <td>0.04</td>
    <td>0</td>
    <td>0.02</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0.06</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
</tbody>
</table>


## Ensembling
Finally, the [Ensemble_kfold-multiple_models.ipynb](3-ensembling/Ensemble_kfold-multiple_models.ipynb) is used to ensemble predictions from the 60 models (4 architectures x 3 hparam configs x 5 folds).

For each model, the top-100 recommended items for all sessions in the test set are saved to a `test_predictions.parquet` file. The ensemble is a weighted sum of the recommendation scores for all models.
