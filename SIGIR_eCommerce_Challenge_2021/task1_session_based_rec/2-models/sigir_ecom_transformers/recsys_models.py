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
import logging

import torch.nn as nn
# load transformer model and its configuration classes
from transformers import (PretrainedConfig, TransfoXLConfig, TransfoXLModel,
                          XLNetConfig, XLNetModel)

logger = logging.getLogger(__name__)


def get_recsys_model(model_args, data_args, training_args, target_size=None):
    total_seq_length = data_args.total_seq_length

    if model_args.model_type == "gru":
        model_cls = nn.GRU(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig(hidden_size=model_args.d_model,)  # dummy config

    elif model_args.model_type == "lstm":
        model_cls = nn.LSTM(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig(hidden_size=model_args.d_model,)

    elif model_args.model_type == "transfoxl":
        model_cls = TransfoXLModel
        config = TransfoXLConfig(
            d_model=model_args.d_model,
            d_embed=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            d_inner=model_args.d_model * 4,
            untie_r=True,
            attn_type=0,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            pad_token_id=data_args.pad_token,
            output_attentions=training_args.log_attention_weights,
            mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
            div_val=1,  # Disables adaptative input (embeddings), because the embeddings are managed by RecSysMetaModel
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    elif model_args.model_type == "xlnet":
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=model_args.d_model,
            d_inner=model_args.d_model * 4,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            ff_activation=model_args.hidden_act,
            untie_r=True,
            bi_data=False,
            attn_type="bi",
            summary_type=model_args.summary_type,
            use_mems_train=True,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            pad_token_id=data_args.pad_token,
            output_attentions=training_args.log_attention_weights,
            mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
            vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
        )

    else:
        raise NotImplementedError

    if model_args.model_type in ["gru", "lstm"]:
        model = model_cls

    elif model_args.model_name_or_path:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_cls(config)

    return model, config
