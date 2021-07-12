#!/bin/bash

mkdir -p ./tmp/

cuda_visible_devices=$1 #GPU devices available for the program, separated by commas. Example: 0,1
fold_number=$2 #Fold number
data_path=$3 # Path of preprocessed data

echo "Running training script"
TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m sigir_ecom_transformers.recsys_main_kfold \
    --output_dir "./tmp/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --validate_every 10 \
    --logging_steps 20 \
    --save_steps 0 \
    --experiments_group "sigir_ecom" \
    --num_folds 5 \
    --bag_number 1 \
    --fold_number ${fold_number} \
    --eval_on_last_item_seq_only \
    --data_path ${data_path} \
    --session_seq_length_max 31 \
    --finetune_on_valid_data \
    ${@:4} # Forwarding Remaining parameters to the script