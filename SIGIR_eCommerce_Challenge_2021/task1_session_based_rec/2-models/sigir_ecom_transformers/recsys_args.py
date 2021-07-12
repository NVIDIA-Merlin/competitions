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
from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_WITH_LM_HEAD_MAPPING
from transformers import TrainingArguments as HfTrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments(HfTrainingArguments):
    validate_every: int = field(
        default=-1,
        metadata={
            "help": "Run validation set every this epoch. "
            "-1 means no validation is used (default: -1)"
        },
    )

    eval_on_test_set: bool = field(
        default=False,
        metadata={
            "help": "Evaluate on test set (by default, evaluates on the validation set)."
        },
    )

    compute_metrics_each_n_steps: int = field(
        default=1,
        metadata={
            "help": "Log metrics each n steps (for train, validation and test sets)"
        },
    )

    log_predictions: bool = field(
        default=False,
        metadata={
            "help": "Logs predictions, labels and metadata features each --compute_metrics_each_n_steps (for test set)."
        },
    )
    log_attention_weights: bool = field(
        default=False,
        metadata={
            "help": "Logs the inputs and attention weights each --compute_metrics_each_n_steps (only test set)"
        },
    )

    experiments_group: str = field(
        default="default",
        metadata={
            "help": "Name of the Experiments Group, for organizing job runs logged on W&B"
        },
    )

    learning_rate_schedule: str = field(
        default="constant_with_warmup",
        metadata={
            "help": "Learning Rate schedule (restarted for each training day). Valid values: constant_with_warmup | linear_with_warmup | cosine_with_warmup"
        },
    )

    learning_rate_warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of steps to linearly increase the learning rate from 0 to the specified initial learning rate schedule. Valid for --learning_rate_schedule = constant_with_warmup | linear_with_warmup | cosine_with_warmup"
        },
    )
    learning_rate_num_cosine_cycles_by_epoch: float = field(
        default=1.25,
        metadata={
            "help": "Number of cycles for by epoch when --learning_rate_schedule = cosine_with_warmup. The number of waves in the cosine schedule (e.g. 0.5 is to just decrease from the max value to 0, following a half-cosine)."
        },
    )

    predict_top_k: int = field(
        default=10,
        metadata={
            "help": "Truncate recommendation list to the highest top-K predicted items (do not affect evaluation metrics computation)"
        },
    )

    eval_steps_on_train_set: int = field(
        default=20,
        metadata={
            "help": "Number of steps to evaluate on train set (which is usually large)"
        },
    )

    max_eval_steps: int = field(
        default=20,
        metadata={"help": "Max evaluation steps (when evaluating loss only)"},
    )

    use_legacy_prediction_loop: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use the legacy prediction_loop in the Trainer."
        },
    )

    # Parameters for SIGIR eCom Data Challenge competition
    train_on_oof_valid_data: bool = field(
        default=False, metadata={"help": "Use OOF valid data to train"}
    )

    save_checkpoint_after_training_stage1: bool = field(
        default=False,
        metadata={
            "help": "Save the model and trainer state to a checkpoint folder under the output folder"
        },
    )

    finetune_on_valid_data: bool = field(
        default=False,
        metadata={
            "help": "Fine tune the model with validation eval data (training on first half of the sessions used for training, the second half is kept for evaluation as the test set)"
        },
    )

    finetune_on_test_data: bool = field(
        default=False,
        metadata={
            "help": "Fine tune the model with test data (training on first half of the sessions which are available)"
        },
    )

    finetune_from_checkpoint_path: str = field(
        default=None,
        metadata={
            "help": "Path of a checkpoint model to fine tune with validation or test data"
        },
    )

    num_folds: int = field(default=5, metadata={"help": "Number of folds for each bag"})
    bag_number: int = field(
        default=1,
        metadata={
            "help": "Number of training bags (each bag will use different seeds)"
        },
    )
    fold_number: int = field(
        default=1, metadata={"help": "Number of folds for each bag"}
    )

    email_submission: Optional[str] = field(
        default="email@domain.com",
        metadata={"help": "E-mail used to name the submission file"},
    )

    do_submit: bool = field(
        default=False, metadata={"help": "Submit to the SIGIR competition"}
    )

    num_epochs_finetuning: int = field(
        default=3, metadata={"help": "Number of finetuning epochs"},
    )

    learning_rate_finetuning: float = field(
        default=0.0001,
        metadata={
            "help": "Learning rate used for fine tuning (Round 2 training). Differently from Round 1 training which uses linear LR decay, fine tuning uses constant LR."
        },
    )

    finetuning_freeze_all_layers_by_item_id_embedding: bool = field(
        default=False,
        metadata={
            "help": "Freeze all layers but the item embedding layer for finetuning"
        },
    )


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default="", metadata={"help": "Path to dataset."},
    )

    pad_token: Optional[int] = field(default=0, metadata={"help": "pad token"})
    mask_token: Optional[int] = field(default=0, metadata={"help": "mask token"})

    # args for selecting which engine to use
    data_loader_engine: Optional[str] = field(
        default="pyarrow",
        metadata={
            "help": "Parquet data loader engine. "
            "'nvtabular': GPU-accelerated parquet data loader from NVTabular, 'pyarrow': read whole parquet into memory. 'petastorm': read chunck by chunck"
        },
    )

    nvt_part_mem_fraction: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Percentage of GPU to allocate for NVTabular dataset / dataloader"
        },
    )

    nvt_part_size: Optional[str] = field(
        default="5MB", metadata={"help": ""},
    )

    feature_config: Optional[str] = field(
        default="config/recsys_input_feature.yaml",
        metadata={
            "help": "yaml file that contains feature information (columns to be read from Parquet file, its dtype, etc)"
        },
    )

    session_seq_length_max: Optional[int] = field(
        default=20,
        metadata={
            "help": "The maximum length of a session (for padding and trimming). For sequential recommendation, this is the maximum length of the sequence"
        },
    )

    # Parameters for SIGIR eCom Data Challenge competition
    use_freq_cap_item_id: bool = field(
        default=False,
        metadata={
            "help": "Uses the encoded item ids with frequency capping (items which occur less than 5 times are encoded as id 1)"
        },
    )

    @property
    def total_seq_length(self) -> int:
        """
        The total sequence length = session length + past session interactions (if --session_aware)
        """
        total_sequence_length = self.session_seq_length_max
        return total_sequence_length


@dataclass
class ModelArguments:

    # args for Hugginface default ones

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    # args for RecSys Meta model
    inp_merge: Optional[str] = field(
        default="mlp", metadata={"help": "input merge mechanism: 'mlp' OR 'attn'"}
    )

    input_features_aggregation: Optional[str] = field(
        default="concat",
        metadata={
            "help": "How input features are merged. Supported options: concat | elementwise_sum_multiply_item_embedding"
        },
    )

    loss_type: Optional[str] = field(
        default="cross_entropy",
        metadata={"help": "Type of Loss function: cross_entropy"},
    )
    model_type: Optional[str] = field(
        default="xlnet",
        metadata={
            "help": "Type of the sequential model. Can be: transfoxl|xlnet"
            + ", ".join(MODEL_TYPES)
        },
    )
    similarity_type: Optional[str] = field(
        default="concat_mlp",
        metadata={
            "help": "how to compute similarity of sequences for negative sampling: 'cosine' OR 'concat_mlp'"
        },
    )
    tf_out_activation: Optional[str] = field(
        default="tanh",
        metadata={"help": "transformer output activation: 'tanh' OR 'relu'"},
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Use Masked Language Modeling (Cloze objective) for training."
        },
    )

    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "Ratio of tokens to mask (set target) from an original sequence. There is a hard constraint that ensures that each sequence have at least one target (masked) and one non-masked item, for effective learning. Thus, if the sequence has more than 2 items, this is the probability of the additional items to be masked"
        },
    )

    # args for Transformers or RNNs

    d_model: Optional[int] = field(
        default=256,
        metadata={
            "help": "size of hidden states (or internal states) for RNNs and Transformers"
        },
    )
    n_layer: Optional[int] = field(
        default=12, metadata={"help": "number of layers for RNNs and Transformers"}
    )
    n_head: Optional[int] = field(
        default=4, metadata={"help": "number of attention heads for Transformers"}
    )
    layer_norm_eps: Optional[float] = field(
        default=1e-12,
        metadata={
            "help": "The epsilon used by the layer normalization layers for Transformers"
        },
    )
    initializer_range: Optional[float] = field(
        default=0.02,
        metadata={
            "help": "The standard deviation of the truncated_normal_initializer for initializing all weight matrices for Transformers"
        },
    )
    hidden_act: Optional[str] = field(
        default="gelu",
        metadata={
            "help": "The non-linear activation function (function or string) in Transformers. 'gelu', 'relu' and 'swish' are supported."
        },
    )
    dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and decoders for Transformers and RNNs"
        },
    )

    input_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The dropout probability of the input embeddings, before being combined with feed-forward layers"
        },
    )

    loss_scale_factor: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Rescale the loss. The scale of different losses types are very different (e.g. cross_entropy > bpr_max > top1_max) and this scaling might help to avoid underflow with fp16"
        },
    )

    attn_type: str = field(
        default="uni",
        metadata={
            "help": "The type of attention. Use 'uni' for Causal LM and 'bi' for Masked LM"
        },
    )

    # args for XLNET
    summary_type: Optional[str] = field(
        default="last",
        metadata={
            "help": "How to summarize the vector representation of the sequence'last', 'first', 'mean', 'attn' are supported"
        },
    )

    eval_on_last_item_seq_only: bool = field(
        default=False,
        metadata={
            "help": "Evaluate metrics only on predictions for the last item of the sequence (rather then evaluation for all next-item predictions)."
        },
    )
    train_on_last_item_seq_only: bool = field(
        default=False,
        metadata={
            "help": "Train only for predicting the last item of the sequence (rather then training to predict for all next-item predictions) (only for Causal LM)."
        },
    )

    use_ohe_item_ids_inputs: bool = field(
        default=False,
        metadata={
            "help": "Uses the one-hot encoding of the item ids as inputs followed by a MLP layer, instead of using item embeddings"
        },
    )

    mf_constrained_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Implements the tying embeddings technique, in which the item id embedding table weights are shared with the last network layer which predicts over all items"
            "This is equivalent of performing a matrix factorization (dot product multiplication) operation between the Transformers output and the item embeddings."
            "This option requires the item id embeddings to have the same dimensions of the last network layer."
        },
    )

    item_embedding_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Dimension of the item embedding. If it is None, a heuristic method used to define the dimension based on items cardinality. "
            "If --mf_constrained_embeddings or --constrained_embeddings are enabled, the output of transformers (dimension defined by --d_model) will "
            "be projected to the same dimension as the item embedding (tying embedding), just before the output layer. "
            "You can define the item embedding dim using --item_embedding_dim or let the size to be defined automatically based on its cardinality multiplied by the --embedding_dim_from_cardinality_multiplier factor."
        },
    )

    features_same_size_item_embedding: bool = field(
        default=False,
        metadata={
            "help": "Makes all features have the same embedding dim than the item embedding."
        },
    )

    stochastic_shared_embeddings_replacement_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of the embedding of a categorical feature be replaced by another an embedding of the same batch"
        },
    )

    softmax_temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature, used to reduce model overconfidence, so that softmax(logits / T). Value 1.0 reduces to regular softmax."
        },
    )

    label_smoothing: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Applies label smoothing using as alpha this parameter value. It helps overconfidence of models and calibration of the predictions."
        },
    )

    embedding_dim_from_cardinality_multiplier: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "Used to define the feature embedding dim based on its cardinality. The formula is embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))"
        },
    )

    item_id_embeddings_init_std: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Uniform distribution maximum and minimun (-bound) value to be used to initialize the item id embedding (usually must be higher than --categs_embeddings_init_uniform_bound, as those weights are also used as the output layer when --mf_constrained_embeddings)"
        },
    )

    other_embeddings_init_std: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Uniform distribution maximum and minimun (-bound) value to be used to initialize the other feature embeddings (other than the item_id, which is defined by --item_id_embeddings_init_uniform_bound)"
        },
    )

    layer_norm_featurewise: bool = field(
        default=False,
        metadata={
            "help": "Enables layer norm for each feature individually, before their aggregation."
        },
    )

    layer_norm_all_features: bool = field(
        default=False,
        metadata={"help": "Enables layer norm after concatenating all features."},
    )

    include_search_features: bool = field(
        default=False,
        metadata={"help": "Uses the search features, leveraging the search module"},
    )

    include_prod_description_emb_feature: bool = field(
        default=False,
        metadata={"help": "Includes the product description vectors as features"},
    )

    include_prod_image_emb_feature: bool = field(
        default=False,
        metadata={"help": "Includes the product image vectors as features"},
    )

    modules_merge: Optional[str] = field(
        default="elementwise",
        metadata={
            "help": "How to merge modules (interactions module, search module, next-item context (event_type)"
        },
    )

    context_event_type_small: bool = field(
        default=False,
        metadata={"help": "Sets late context to use the small event type embedding"},
    )

    search_transformers_dmodel: Optional[int] = field(
        default=100,
        metadata={
            "help": "Hidden size of transformer architectures for the search model"
        },
    )

