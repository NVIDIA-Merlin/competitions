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
import json
import logging
import os
import pickle
import random
import sys
import time
import types

import dllogger as DLLogger
import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from dllogger import Verbosity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import is_main_process

from .recsys_args import DataArguments, ModelArguments, TrainingArguments
from .recsys_data import fetch_data_loader
from .recsys_meta_model import RecSysMetaModel
from .recsys_models import get_recsys_model
from .recsys_outputs import (
    AttentionWeightsLogger,
    PredictionLogger,
    config_dllogger,
    creates_output_dir,
    log_aot_metric_results,
    log_parameters,
    set_log_attention_weights_callback,
)
from .recsys_trainer import DatasetType, RecSysTrainer
from .recsys_utils import get_label_feature_name
from .sigir_ecom_challenge_code.evaluation_sigir import (
    next_item_metric,
    subsequent_items_metric,
)
from .sigir_ecom_challenge_code.submission.uploader import upload_submission

logger = logging.getLogger(__name__)


# this code use Version 3
assert sys.version_info.major > 2


def main():

    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (data_args, model_args, training_args,) = parser.parse_args_into_dataclasses()

    # Ensuring to set W&B run name to null, so that a nice run name is generated
    training_args.run_name = None

    # Loading features config file
    with open(data_args.feature_config) as yaml_file:
        feature_map = yaml.load(yaml_file, Loader=yaml.FullLoader)

    label_name = get_label_feature_name(feature_map)
    target_size = feature_map[label_name]["cardinality"]

    creates_output_dir(training_args)

    bag_number = training_args.bag_number
    fold = training_args.fold_number
    output_bag_folder = os.path.join(training_args.output_dir, f"bag_{bag_number}")
    output_fold_folder = os.path.join(output_bag_folder, f"fold_{fold}")
    os.makedirs(output_fold_folder, exist_ok=True)

    config_logging(training_args)

    (
        product_url_to_emb_pos,
        emb_pos_to_product_url,
        description_embedding_matrix,
        image_embedding_matrix,
    ) = load_prod_embedding_matrices(data_args.data_path, target_size)
    prod_embeddings_dict = {
        "product_url_to_emb_pos": product_url_to_emb_pos,
        "emb_pos_to_product_url": emb_pos_to_product_url,
        "description_embedding_matrix": description_embedding_matrix,
        "image_embedding_matrix": image_embedding_matrix,
    }

    # Instantiates the model defined in --model_type
    seq_model, config = get_recsys_model(
        model_args, data_args, training_args, target_size
    )
    # Instantiate the RecSys Meta Model
    rec_model = RecSysMetaModel(
        seq_model, config, model_args, data_args, feature_map, prod_embeddings_dict
    )

    # Instantiate the RecSysTrainer, which manages training and evaluation
    trainer = RecSysTrainer(
        model=rec_model, args=training_args, model_args=model_args, data_args=data_args,
    )

    log_parameters(trainer, data_args, model_args, training_args)

    set_log_attention_weights_callback(trainer, training_args)

    metrics_results = {}

    ################# SIGIR eCom Data Challenge - k-fold train #################

    logger.info(f"Starting BAG {bag_number} - FOLD {fold}")

    DLLogger.log(
        step="PARAMETER",
        data={"bag_number": bag_number, "fold_number": fold},
        verbosity=Verbosity.DEFAULT,
    )

    set_seed((bag_number * 10) + fold)

    data_path = data_args.data_path

    freqcap_suffix = ""
    if data_args.use_freq_cap_item_id:
        freqcap_suffix = "-freqcap"

    valid_parquet_file = os.path.join(
        data_path, f"valid-eval{freqcap_suffix}-{fold}.parquet"
    )
    valid_labels_parquet_file = os.path.join(
        data_path, f"valid-eval-labels{freqcap_suffix}-{fold}.parquet"
    )
    test_full_parquet_file = os.path.join(
        data_path, f"test-full{freqcap_suffix}.parquet"
    )

    training_datasets = [f"train{freqcap_suffix}"]
    if training_args.train_on_oof_valid_data:
        training_datasets += [f"valid-train{freqcap_suffix}"]

    training_files = []

    # Defining the files for 1st training round
    for dataset in training_datasets:
        for f in range(1, training_args.num_folds + 1):
            if f != fold:
                training_files.append(os.path.join(data_path, f"{dataset}-{f}.parquet"))

    random.shuffle(training_files)

    finetuning_files = []
    # Defining the files for 2nd training round (fine-tuning)
    for f in range(1, training_args.num_folds + 1):
        if f != fold:
            if training_args.finetune_on_test_data:
                finetuning_files.append(
                    os.path.join(data_path, f"test{freqcap_suffix}-{f}.parquet")
                )
            elif training_args.finetune_on_valid_data:
                finetuning_files.append(
                    os.path.join(data_path, f"valid-eval{freqcap_suffix}-{f}.parquet")
                )

    random.shuffle(finetuning_files)

    logger.info(f"Train (Round #1) parquet files: {training_files}")
    logger.info(f"Train (Round #2 - Fine tuning) parquet files: {finetuning_files}")

    logger.info(f"Valid parquet file: {valid_parquet_file}")
    logger.info(f"Test parquet file: {test_full_parquet_file}")

    train_loader = fetch_data_loader(
        data_args,
        training_args,
        feature_map,
        training_files,
        is_train_set=True,
        shuffle_dataloader=True,
    )
    eval_loader = fetch_data_loader(
        data_args, training_args, feature_map, [valid_parquet_file], is_train_set=False,
    )

    mapping_product_sku_without_urls_df = pd.read_parquet(
        os.path.join(data_path, "categories/mapping_product_sku_without_urls.parquet")
    )
    item_id_mapping = dict(
        zip(
            mapping_product_sku_without_urls_df["encoded_product_sku"].values,
            mapping_product_sku_without_urls_df["original_product_sku"].values,
        )
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        logger.info(f"************* Training (Round #1) *************")

        trainer.set_train_dataloader(train_loader)
        trainer.set_eval_dataloader(eval_loader)

        trainer.prediction_loss_only = False
        trainer.reset_lr_scheduler()

        if training_args.finetune_from_checkpoint_path is None:
            trainer.train(model_path=model_path)

            if training_args.save_checkpoint_after_training_stage1:
                logger.info(
                    f"************* Saving model before fine tuning *************"
                )
                logger.info("Saving model...")
                trainer._save_checkpoint(rec_model, trial=None, metrics=None)

        if len(finetuning_files) > 0:
            logger.info(f"************* Fine-Tuning (Round #2) *************")

            # Fine tuning with test set
            train_loader = fetch_data_loader(
                data_args,
                training_args,
                feature_map,
                finetuning_files,
                is_train_set=True,
                shuffle_dataloader=True,
            )

            if training_args.finetune_from_checkpoint_path is not None:
                load_model_trainer_states_from_checkpoint(
                    training_args.finetune_from_checkpoint_path, rec_model, trainer
                )

            # Adjusting finetuning-specific training params
            training_args.num_train_epochs = training_args.num_epochs_finetuning
            training_args.learning_rate = training_args.learning_rate_finetuning

            if training_args.finetuning_freeze_all_layers_by_item_id_embedding:
                for n, v in rec_model.named_parameters():
                    v.requires_grad = n in [
                        "embedding_tables.product_id.weight",
                        "output_layer_bias",
                    ]

            # training_args.learning_rate_schedule = "constant_with_warmup"
            trainer.reset_optimizer()
            trainer.reset_lr_scheduler()

            trainer.set_train_dataloader(train_loader)
            trainer.train(model_path=model_path)

    # Evaluation
    if training_args.do_eval:
        logger.info(f"************* Evaluation *************")

        eval_loader_with_dummy_session_end = fetch_data_loader(
            data_args,
            training_args,
            feature_map,
            [valid_parquet_file],
            is_train_set=False,
            shuffle_dataloader=False,
            add_dummy_item_end_sequence_label_column=True,
            label_column_name=label_name,
        )

        trainer.set_test_dataloader(eval_loader_with_dummy_session_end)
        # trainer.should_compute_metrics = False
        pred_outputs = trainer.predict(None, metric_key_prefix=DatasetType.test.value)
        eval_preds_items = pred_outputs.predictions[0]
        eval_pred_logits = pred_outputs.predictions[1].astype(np.float32)
        print("eval_preds.shape", eval_preds_items.shape)

        valid_eval_df = pd.read_parquet(valid_parquet_file)[: len(eval_preds_items)]

        eval_sessions_items = valid_eval_df["product_url_hash_list"]

        # Removing predictions on hashed_urls (page views), keeping only predictions on product skus
        # and also removing predictions on items that appear in the first half of the sessions
        eval_preds_items, eval_pred_logits = filter_predictions(
            eval_sessions_items.tolist(),
            eval_preds_items.tolist(),
            eval_pred_logits.tolist(),
            item_id_mapping,
        )

        valid_eval_df["pred_item_ids"] = eval_preds_items
        valid_eval_df["pred_item_logits"] = eval_pred_logits

        if not training_args.finetune_on_test_data:
            # Save predictions to parquet (intermediate representation)
            valid_eval_df.to_parquet(
                os.path.join(output_fold_folder, "valid_eval_predictions.parquet")
            )

        valid_eval_labels_df = pd.read_parquet(valid_labels_parquet_file)
        eval_labels = valid_eval_labels_df["labels"].values[: len(eval_preds_items)]
        print("eval_labels.shape", eval_labels.shape)

        mrr_result = next_item_metric(eval_preds_items, eval_labels, top_K=20)
        f1_results = subsequent_items_metric(eval_preds_items, eval_labels, top_K=20)

        metrics_results = {"mrr@20": mrr_result, "f1@20": f1_results}
        logger.info("Evaluation metrics: {}".format(metrics_results))

    if training_args.do_predict:
        logger.info(f"************* Predicting for the test set *************")

        test_loader_with_dummy_session_end = fetch_data_loader(
            data_args,
            training_args,
            feature_map,
            [test_full_parquet_file],
            is_train_set=False,
            shuffle_dataloader=False,
            add_dummy_item_end_sequence_label_column=True,
            label_column_name=label_name,
        )

        # Predicting over the full test set
        trainer.set_test_dataloader(test_loader_with_dummy_session_end)
        # trainer.should_compute_metrics = False
        pred_outputs = trainer.predict(None, metric_key_prefix=DatasetType.test.value)
        test_preds_items = pred_outputs.predictions[0]
        test_pred_logits = pred_outputs.predictions[1].astype(np.float32)

        # Combining predictions with session hash ids
        test_predictions_df = pd.read_parquet(test_full_parquet_file)[
            ["original_session_id_hash", "session_id_hash", "product_url_hash_list"]
        ][: test_preds_items.shape[0]]

        test_sessions_items = test_predictions_df["product_url_hash_list"]

        # Removing predictions on hashed_urls (page views), keeping only predictions on product skus
        # and also removing predictions on items that appear in the first half of the sessions
        test_preds_items, test_pred_logits = filter_predictions(
            test_sessions_items.tolist(),
            test_preds_items.tolist(),
            test_pred_logits.tolist(),
            item_id_mapping,
        )

        test_predictions_df["pred_item_ids"] = test_preds_items
        test_predictions_df["pred_item_logits"] = test_pred_logits
        # Save predictions to parquet (intermediate representation)
        test_predictions_df.to_parquet(
            os.path.join(output_fold_folder, "test_predictions.parquet")
        )

    if training_args.do_submit:

        logger.info(f"************* Generating the submission file *************")
        test_predictions_df = pd.read_parquet(
            os.path.join(output_fold_folder, "test_predictions.parquet")
        )
        local_prediction_file_path = generate_submission_file(
            test_predictions_df,
            item_id_mapping,
            data_path,
            training_args,
            output_fold_folder,
        )

        logger.info(f"************* Uploading the submission file *************")
        TASK = "rec"  # 'rec' or 'cart'
        upload_submission(local_file=local_prediction_file_path, task=TASK)

    logger.info("Training and evaluation loops are finished")

    if trainer.is_world_process_zero():

        """
        logger.info("Saving model...")
        trainer.save_model()
        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
        """

        if training_args.do_eval:
            logger.info("Computing and loging metrics")
            results_df = pd.DataFrame([metrics_results])
            results_df.to_csv(
                os.path.join(output_fold_folder, "eval_train_results.csv"), index=False,
            )

            # Computing average metrics
            metrics_results_log = {f"{k}_final": v for k, v in metrics_results.items()}
            # Logging to W&B
            trainer.log(metrics_results_log)

            log_aot_metric_results(output_fold_folder, metrics_results_log)


def filter_predictions(
    sessions_items, sessions_pred_items, sessions_pred_logits, item_id_mapping
):
    missing_index = list(item_id_mapping.values()).index("missing")
    missing_item_id = list(item_id_mapping.keys())[missing_index]

    UNFREQ_ITEM_ID = 1

    # Removing predictions on hashed_urls (page views), keeping only predictions on product skus
    # and also removing predictions on items that appear in the first half of the sessions

    filtered_sessions_pred_items = []
    filtered_sessions_pred_logits = []
    for session_items, session_pred_items, session_pred_logits in zip(
        sessions_items, sessions_pred_items, sessions_pred_logits
    ):
        current_session_pred_items = []
        current_session_pred_logits = []
        for item_id, pred_logit in zip(session_pred_items, session_pred_logits):
            if (
                item_id in item_id_mapping
                and item_id not in session_items
                and item_id != missing_item_id
                and item_id != UNFREQ_ITEM_ID
            ):
                current_session_pred_items.append(item_id)
                current_session_pred_logits.append(pred_logit)

        filtered_sessions_pred_items.append(current_session_pred_items)
        filtered_sessions_pred_logits.append(current_session_pred_logits)

    return filtered_sessions_pred_items, filtered_sessions_pred_logits


def load_model_trainer_states_from_checkpoint(checkpoint_path, rec_model, trainer):
    """
    This method loads from checkpoints states of the model, trainer and random states. 
    It does not loads the optimizer and LR scheduler states (for that call trainer.train() with resume_from_checkpoint argument for a complete load)
    """
    logger.info("Loading previously trained model")
    # Restoring model weights
    rec_model.load_state_dict(
        # torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
        torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
    )

    # Restoring random state
    rng_file = os.path.join(checkpoint_path, "rng_state.pth")
    checkpoint_rng_state = torch.load(rng_file)
    random.setstate(checkpoint_rng_state["python"])
    np.random.set_state(checkpoint_rng_state["numpy"])
    torch.random.set_rng_state(checkpoint_rng_state["cpu"])
    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
    # Restoring AMP scaler
    trainer.scaler.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "scaler.pt"))
    )


def generate_submission_file(
    test_predictions_df, item_id_mapping, data_path, training_args, output_fold_folder
):

    test_predictions_df = test_predictions_df.set_index("original_session_id_hash")

    with open(os.path.join(data_path, "rec_test_phase_2.json")) as json_file:
        test_queries = json.load(json_file)

    assert len(test_predictions_df) == len(test_queries)

    sessions_not_found = 0
    preds_with_length_less_than_20 = 0
    count = 0
    sessions_lengths_list = []
    for session in tqdm(test_queries):
        session_id_hash = session["query"][0]["session_id_hash"]

        session_product_interacted = set(
            list(
                [
                    interaction["product_sku_hash"]
                    for interaction in session["query"]
                    if interaction["product_sku_hash"] is not None
                ]
            )
        )

        if session_id_hash in test_predictions_df.index:
            predictions = test_predictions_df.loc[session_id_hash]["pred_item_ids"]

            # Converting to the original product sku
            predictions = list(
                [
                    item_id_mapping[pred_item]
                    for pred_item in predictions
                    if item_id_mapping[pred_item] != "missing"
                ]
            )

            # assert len(set(predictions).intersection(session_product_interacted)) == 0

            # Removing from predictions any item that was already interacted within the session
            predictions = [
                p for p in predictions if p not in session_product_interacted
            ]

            if len(predictions) < 20:
                preds_with_length_less_than_20 += 1
                # logger.warn(
                #    "Predictions for session {} should be 20 but are {}.".format(
                #        session_id_hash, len(predictions)
                #    )
                # )

        else:
            raise Exception(
                "Session {} not found in the preprocessed test set".format(session)
            )

        sessions_lengths_list.append(len(predictions))

        count += 1

        session["label"] = predictions[:20]

    total_sessions = len(test_queries)
    logger.warn(
        f"# Total sessions: {total_sessions} - # Sessions not found: {sessions_not_found} - # Sessions with length < 20: {preds_with_length_less_than_20} - avg length: "
        + str(sum(sessions_lengths_list) / len(sessions_lengths_list))
    )

    local_prediction_file = "{}_{}.json".format(
        training_args.email_submission.replace("@", "_"), round(time.time() * 1000)
    )
    local_prediction_file_path = os.path.join(output_fold_folder, local_prediction_file)
    logger.info("Generating JSON file with predictions")
    with open(local_prediction_file_path, "w") as fp:
        json.dump(test_queries, fp, indent=2)

    return local_prediction_file_path


def load_prod_embedding_matrices(path, target_size):
    with open(os.path.join(path, "embedding_data.pkl"), "rb") as infile:
        embedding_data = pickle.load(infile)

    description_embedding_matrix = embedding_data[0].astype(np.float32)
    image_embedding_matrix = embedding_data[1].astype(np.float32)
    product_url_encoded_to_embedding_pos_dict = embedding_data[2]

    # Applying L2 norm to the vectors
    description_embedding_matrix = normalize(
        description_embedding_matrix, norm="l2", axis=1
    )
    image_embedding_matrix = normalize(image_embedding_matrix, norm="l2", axis=1)

    # Creating a tensor to map the encoded product urls to the position in the product embeddings
    product_url_to_emb_pos = np.zeros(target_size, dtype=np.int64)
    product_url_to_emb_pos[
        list(product_url_encoded_to_embedding_pos_dict.keys())
    ] = list(product_url_encoded_to_embedding_pos_dict.values())

    # Creating a tensor to map the encoded product urls to the position in the product embeddings
    emb_pos_to_product_url = np.zeros(
        len(product_url_encoded_to_embedding_pos_dict), dtype=np.int64
    )
    emb_pos_to_product_url[
        list(product_url_encoded_to_embedding_pos_dict.values())
    ] = list(product_url_encoded_to_embedding_pos_dict.keys())

    return (
        product_url_to_emb_pos,
        emb_pos_to_product_url,
        description_embedding_matrix,
        image_embedding_matrix,
    )


def config_logging(training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    config_dllogger(training_args.output_dir)


if __name__ == "__main__":
    main()

