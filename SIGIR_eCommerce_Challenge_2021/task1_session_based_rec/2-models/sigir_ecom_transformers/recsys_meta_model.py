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
"""
A meta class supports various (Huggingface) transformer models for RecSys tasks.
"""

import logging
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from transformers import PreTrainedModel, XLNetConfig, XLNetModel

logger = logging.getLogger(__name__)

torch.manual_seed(0)


class ProjectionNetwork(nn.Module):
    """
    Project item interaction embeddings into model's hidden size
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        inp_merge,
        layer_norm_all_features,
        input_dropout,
        tf_out_act,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inp_merge = inp_merge
        self.layer_norm_all_features = layer_norm_all_features
        self.input_dropout = input_dropout
        self.tf_out_act = tf_out_act
        self.input_dropout = nn.Dropout(self.input_dropout)
        if self.layer_norm_all_features:
            self.layer_norm_all_input = nn.LayerNorm(normalized_shape=self.input_dim)
        if self.inp_merge == "mlp":
            self.merge = nn.Linear(self.input_dim, output_dim)

        elif self.inp_merge == "identity":
            assert self.input_dim == self.output_dim, (
                "Input dim '%s' should be equal to the model's hidden size '%s' when inp_merge=='identity'"
                % (self.input_dim, self.output_dim)
            )
            self.merge = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, inp):
        if self.inp_merge == "mlp" and self.layer_norm_all_features:
            return self.tf_out_act(
                self.merge(self.layer_norm_all_input(self.input_dropout(inp)))
            )
        elif self.inp_merge == "mlp":
            return self.tf_out_act(self.merge(self.input_dropout(inp)))
        return self.merge(inp)


PRODUCT_QUERY_EMBEDDINGS_SIZE = 50


class SearchModule(nn.Module):
    def __init__(
        self,
        model_args,
        data_args,
        # products_embedding_table,
        item_embedding_dim,
        d_model=100,
    ):
        super().__init__()

        """
        self.products_embedding_table = products_embedding_table
        """

        self.empty_query_vector = torch.normal(
            mean=0.0, std=0.07, size=(PRODUCT_QUERY_EMBEDDINGS_SIZE,),
        )

        # Search features
        self.ln_flat_query_vector_trunc = nn.LayerNorm(
            normalized_shape=PRODUCT_QUERY_EMBEDDINGS_SIZE
        )

        self.ln_impressions_size_trunc_norm = nn.LayerNorm(normalized_shape=1)
        self.ln_clicks_size_trunc_norm = nn.LayerNorm(normalized_shape=1)

        """
        self.ln_flat_clicked_skus_hash_trunc = nn.LayerNorm(
            normalized_shape=item_embedding_dim
        )
        """

        self.merge_queries_input = ProjectionNetwork(
            PRODUCT_QUERY_EMBEDDINGS_SIZE,
            model_args.d_model,
            model_args.inp_merge,
            False,
            model_args.input_dropout,
            torch.tanh,
        )

        """
        self.merge_queries_input = ProjectionNetwork(
            PRODUCT_QUERY_EMBEDDINGS_SIZE + 2,
            d_model,
            model_args.inp_merge,
            False,
            model_args.input_dropout,
            torch.nn.GELU(),
        )

        self.merge_clicks_input = ProjectionNetwork(
            item_embedding_dim,
            d_model,
            model_args.inp_merge,
            False,
            model_args.input_dropout,
            torch.nn.GELU(),
        )
        """

        self.search_module_queries = XLNetModel(
            XLNetConfig(
                d_model=d_model,
                d_inner=d_model * 4,
                n_layer=2,
                n_head=4,
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
                output_attentions=False,
                mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
                vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
            )
        )

        """
        self.search_module_clicks = XLNetModel(
            XLNetConfig(
                d_model=d_model,
                d_inner=d_model * 4,
                n_layer=2,
                n_head=4,
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
                output_attentions=False,
                mem_len=1,  # We do not use mems, because we feed the full sequence to the Transformer models and not sliding segments (which is useful for the long sequences in NLP. As setting mem_len to 0 leads to NaN in loss, we set it to one, to minimize the computing overhead)
                vocab_size=1,  # As the input_embeds will be fed in the forward function, limits the memory reserved by the internal input embedding table, which will not be used
            )
        )
        """

    def forward(self, search_inputs):
        search_inputs_processed = self.feature_process_search_features(search_inputs)
        """
        search_queries_seq = self.merge_features(search_inputs_processed)

        # Search queries sub-network
        search_queries_output = self.search_module_queries(
            inputs_embeds=search_queries_seq
        )
        search_queries_output = search_queries_output[0]
        # Using the first output to represent the sequence
        search_queries_output = search_queries_output[:, 0:1]
        # Average pooling of outputs
        # search_queries_output = search_queries_output.mean(axis=1, keepdims=True)
        """

        """
        # Search clicks sub-network
        search_clicks_output = self.search_module_clicks(
            inputs_embeds=search_clicks_seq
        )
        search_clicks_output = search_clicks_output[0]
        # Using the first output to represent the sequence
        search_clicks_output = search_clicks_output[:, 0:1]
        # Average pooling of outputs
        # search_clicks_output = search_clicks_output.mean(axis=1, keepdims=True)
        """

        # Averaging
        search_queries_output = search_inputs_processed["flat_query_vector_trunc"].mean(
            axis=1, keepdims=True
        )
        search_queries_output = self.merge_queries_input(search_queries_output)

        return search_queries_output  # , search_clicks_output

    def feature_process_search_features(self, search_inputs):
        transformed_features = OrderedDict()

        """
        transformed_features[
            "flat_clicked_skus_hash_trunc"
        ] = self.products_embedding_table(search_inputs["flat_clicked_skus_hash_trunc"])
        transformed_features[
            "flat_clicked_skus_hash_trunc"
        ] = self.ln_flat_clicked_skus_hash_trunc(
            transformed_features["flat_clicked_skus_hash_trunc"]
        )
        """

        transformed_features["clicks_size_trunc_norm"] = search_inputs[
            "clicks_size_trunc_norm"
        ].unsqueeze(-1)
        transformed_features["clicks_size_trunc_norm"] = self.ln_clicks_size_trunc_norm(
            transformed_features["clicks_size_trunc_norm"]
        )

        transformed_features["impressions_size_trunc_norm"] = search_inputs[
            "impressions_size_trunc_norm"
        ].unsqueeze(-1)
        transformed_features[
            "impressions_size_trunc_norm"
        ] = self.ln_impressions_size_trunc_norm(
            transformed_features["impressions_size_trunc_norm"]
        )

        transformed_features["flat_query_vector_trunc"] = search_inputs[
            "flat_query_vector_trunc"
        ].reshape(
            list(search_inputs["impressions_size_trunc_norm"].shape)
            + [PRODUCT_QUERY_EMBEDDINGS_SIZE]
        )

        # Applying L2 norm in the query vector
        transformed_features["flat_query_vector_trunc"] = torch.nn.functional.normalize(
            transformed_features["flat_query_vector_trunc"]
        )

        # Replacing zeroed query vectors by an embedding
        empty_query_mask = (
            transformed_features["flat_query_vector_trunc"].max(axis=-1).values == 0.0
        )
        transformed_features["flat_query_vector_trunc"][
            empty_query_mask
        ] = self.empty_query_vector.to(
            transformed_features["flat_query_vector_trunc"].device
        )

        """
        #Applying layer norm
        transformed_features[
            "flat_query_vector_trunc"
        ] = self.ln_flat_query_vector_trunc(
            transformed_features["flat_query_vector_trunc"]
        )
        """

        # Truncating to the first query vectors and clicks
        """
        transformed_features["flat_clicked_skus_hash_trunc"] = transformed_features[
            "flat_clicked_skus_hash_trunc"
        ][:, :2]
        """
        transformed_features["impressions_size_trunc_norm"] = transformed_features[
            "impressions_size_trunc_norm"
        ][:, :2]
        transformed_features["clicks_size_trunc_norm"] = transformed_features[
            "clicks_size_trunc_norm"
        ][:, :2]
        transformed_features["flat_query_vector_trunc"] = transformed_features[
            "flat_query_vector_trunc"
        ][:, :2]

        return transformed_features

    def merge_features(self, search_inputs):
        search_seq = torch.cat(
            [
                search_inputs["flat_query_vector_trunc"],
                search_inputs["impressions_size_trunc_norm"],
                search_inputs["clicks_size_trunc_norm"],
            ],
            axis=-1,
        )
        search_queries_seq = self.merge_queries_input(search_seq)

        """
        search_clicks_seq = search_inputs["flat_clicked_skus_hash_trunc"]
        search_clicks_seq = self.merge_clicks_input(search_clicks_seq)
        """

        return search_queries_seq  # , search_clicks_seq


def get_embedding_size_from_cardinality(cardinality, multiplier=2.0):
    # A rule-of-thumb from Google.
    embedding_size = int(math.ceil(math.pow(cardinality, 0.25) * multiplier))
    return embedding_size


class RecSysMetaModel(PreTrainedModel):
    """
    vocab_sizes : sizes of vocab for each discrete inputs
        e.g., [product_id_vocabs, category_vocabs, etc.]
    """

    def __init__(
        self, model, config, model_args, data_args, feature_map, prod_embeddings_dict
    ):
        super(RecSysMetaModel, self).__init__(config)

        self.include_search_features = model_args.include_search_features
        self.include_prod_description_emb_feature = (
            model_args.include_prod_description_emb_feature
        )
        self.include_prod_image_emb_feature = model_args.include_prod_image_emb_feature
        self.modules_merge = model_args.modules_merge
        self.context_event_type_small = model_args.context_event_type_small
        self.use_freq_cap_item_id = data_args.use_freq_cap_item_id

        self.layer_norm_featurewise = model_args.layer_norm_featurewise
        self.layer_norm_all_features = model_args.layer_norm_all_features

        self.items_ids_sorted_by_freq = None
        self.neg_samples = None

        self.model = model

        self.feature_map = feature_map

        self.pad_token = data_args.pad_token
        self.mask_token = data_args.mask_token

        self.use_ohe_item_ids_inputs = model_args.use_ohe_item_ids_inputs
        self.stochastic_shared_embeddings_replacement_prob = (
            model_args.stochastic_shared_embeddings_replacement_prob
        )

        self.loss_scale_factor = model_args.loss_scale_factor
        self.softmax_temperature = model_args.softmax_temperature
        self.label_smoothing = model_args.label_smoothing

        self.mf_constrained_embeddings = model_args.mf_constrained_embeddings
        self.item_id_embeddings_init_std = model_args.item_id_embeddings_init_std
        self.other_embeddings_init_std = model_args.other_embeddings_init_std

        self.item_embedding_dim = model_args.item_embedding_dim
        self.features_same_size_item_embedding = (
            model_args.features_same_size_item_embedding
        )
        self.embedding_dim_from_cardinality_multiplier = (
            model_args.embedding_dim_from_cardinality_multiplier
        )

        self.input_features_aggregation = model_args.input_features_aggregation

        self.define_features_layers(model_args)
        # Adding the dim of the product description and image embeddings
        if self.include_prod_description_emb_feature:
            self.input_combined_dim += PRODUCT_QUERY_EMBEDDINGS_SIZE
        if self.include_prod_image_emb_feature:
            self.input_combined_dim += PRODUCT_QUERY_EMBEDDINGS_SIZE

        self.prod_embeddings_dict = prod_embeddings_dict
        self.define_product_description_image_embeddings()

        if self.include_search_features:
            self.search_module = SearchModule(
                model_args,
                data_args,
                item_embedding_dim=self.item_embedding_dim,
                d_model=model_args.search_transformers_dmodel,
            )

        if self.layer_norm_featurewise:
            # Product description and image embeddings
            self.features_layer_norm["prod_description_embedding"] = nn.LayerNorm(
                normalized_shape=PRODUCT_QUERY_EMBEDDINGS_SIZE
            )

            self.features_layer_norm["prod_image_embedding"] = nn.LayerNorm(
                normalized_shape=PRODUCT_QUERY_EMBEDDINGS_SIZE
            )

        self.inp_merge = model_args.inp_merge

        if model_args.tf_out_activation == "tanh":
            self.tf_out_act = torch.tanh
        elif model_args.tf_out_activation == "relu":
            self.tf_out_act = torch.relu

        self.merge = ProjectionNetwork(
            self.input_combined_dim,
            config.hidden_size,
            self.inp_merge,
            self.layer_norm_all_features,
            model_args.input_dropout,
            self.tf_out_act,
        )

        if self.modules_merge in ["concat", "elementwise_context_concat_search"]:
            input_dim = (
                model_args.d_model
            )  # Transformers output + context features(event_type_large)
            if self.modules_merge == "concat":
                if self.context_event_type_small:
                    input_dim += self.embedding_tables["event_type"].weight.shape[1]
                else:
                    input_dim += config.hidden_size
            if self.include_search_features:
                input_dim += model_args.search_transformers_dmodel  # search_queries_seq

            self.modules_merge_fc = ProjectionNetwork(
                input_dim=input_dim,
                output_dim=config.hidden_size,
                inp_merge=self.inp_merge,
                layer_norm_all_features=False,
                input_dropout=model_args.input_dropout,
                tf_out_act=torch.nn.GELU(),
            )

        if not self.context_event_type_small:
            self.embedding_tables["event_type_large"] = nn.Embedding(
                self.feature_map["event_type-list"]["cardinality"],
                config.hidden_size,
                # self.item_embedding_dim,
                padding_idx=self.pad_token,
            ).to(self.device)

            with torch.no_grad():
                self.embedding_tables["event_type_large"].weight.normal_(
                    0.0, self.other_embeddings_init_std
                )

        if self.use_freq_cap_item_id:
            self.embedding_tables["product_url_hash_list_unfreq_large"] = nn.Embedding(
                self.feature_map["product_url_hash_list_unfreq"]["cardinality"],
                config.hidden_size,
                # self.item_embedding_dim,
                padding_idx=self.pad_token,
            ).to(self.device)

            with torch.no_grad():
                self.embedding_tables[
                    "product_url_hash_list_unfreq_large"
                ].weight.normal_(0.0, self.other_embeddings_init_std)

        self.eval_on_last_item_seq_only = model_args.eval_on_last_item_seq_only
        self.train_on_last_item_seq_only = model_args.train_on_last_item_seq_only

        self.n_layer = model_args.n_layer

        # Args for Masked-LM task
        self.mlm = model_args.mlm
        self.mlm_probability = model_args.mlm_probability

        # Creating a trainable embedding for masking inputs for Masked LM
        self.masked_item_embedding = nn.Parameter(torch.Tensor(config.hidden_size)).to(
            self.device
        )
        nn.init.normal_(
            self.masked_item_embedding, mean=0, std=0.001,
        )

        self.similarity_type = model_args.similarity_type

        self.output_layer = nn.Linear(config.hidden_size, self.target_dim).to(
            self.device
        )

        self.loss_type = model_args.loss_type
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.output_layer_bias = nn.Parameter(torch.Tensor(self.target_dim)).to(
            self.device
        )
        nn.init.zeros_(self.output_layer_bias)

        if self.label_smoothing > 0.0:
            self.loss_nll = LabelSmoothCrossEntropyLoss(smoothing=self.label_smoothing)
        else:
            self.loss_nll = nn.NLLLoss(ignore_index=self.pad_token)

        tf_out_size = model_args.d_model

        if model_args.mf_constrained_embeddings:
            transformer_output_projection_dim = self.item_embedding_dim

        else:
            transformer_output_projection_dim = config.hidden_size

        self.transformer_output_project = nn.Linear(
            tf_out_size, transformer_output_projection_dim
        ).to(self.device)

    def forward(self, *args, **kwargs):
        inputs = kwargs

        # Step1. Unpack inputs, get embedding, and concatenate them
        label_seq = None
        (pos_inp, label_seq, metadata_for_pred_logging) = self.feature_process(inputs)
        assert label_seq is not None, "label sequence is not declared in feature_map"

        if self.include_search_features:
            search_queries_output = self.search_module(inputs)

        if (
            self.include_prod_description_emb_feature
            or self.include_prod_image_emb_feature
        ):

            if (
                self.prod_embeddings_dict["product_url_to_emb_pos"].device
                != self.device
            ):
                # Moving embeddings to the device
                for k, v in self.prod_embeddings_dict.items():
                    self.prod_embeddings_dict[k] = v.to(self.device)

            # Getting the embeddings of the product description and image
            product_url_embedding_idxs = self.prod_embeddings_dict[
                "product_url_to_emb_pos"
            ][inputs["product_url_hash_list"]]

            if self.include_prod_description_emb_feature:

                product_description_embeddings = self.prod_embeddings_dict[
                    "description_embedding_matrix"
                ][product_url_embedding_idxs]

                # Concatenating the product embeddings with the other features
                pos_inp = torch.cat([pos_inp, product_description_embeddings], axis=-1)

            if self.include_prod_image_emb_feature:
                product_image_embeddings = self.features_layer_norm[
                    "prod_image_embedding"
                ](
                    self.prod_embeddings_dict["image_embedding_matrix"][
                        product_url_embedding_idxs
                    ]
                )

                pos_inp = torch.cat([pos_inp, product_image_embeddings], axis=-1)

        if self.context_event_type_small:
            post_fusion_context_vector = self.embedding_tables["event_type"](
                inputs["event_type-list"]
            )
        else:
            post_fusion_context_vector = self.embedding_tables["event_type_large"](
                inputs["event_type-list"]
            )

            if self.use_freq_cap_item_id:
                post_fusion_context_vector = post_fusion_context_vector + self.embedding_tables[
                    "product_url_hash_list_unfreq_large"
                ](
                    inputs["product_url_hash_list_unfreq"]
                )

        if self.include_search_features:

            if self.modules_merge in ["elementwise", "elementwise_sum"]:
                if self.include_search_features:
                    post_fusion_context_vector = (
                        post_fusion_context_vector
                        + search_queries_output
                        # + search_clicks_output
                    )
            elif self.modules_merge == "concat":
                post_fusion_context_vector = torch.cat(
                    [
                        post_fusion_context_vector,
                        search_queries_output.repeat(
                            1, post_fusion_context_vector.shape[1], 1
                        ),
                        # search_clicks_output.repeat(
                        #    1, post_fusion_context_vector.shape[1], 1
                        # ),
                    ],
                    dim=-1,
                )
            elif self.modules_merge == "elementwise_context_concat_search":
                search_outputs_concat = torch.cat(
                    [
                        search_queries_output.repeat(
                            1, post_fusion_context_vector.shape[1], 1
                        ),
                        # search_clicks_output.repeat(
                        #    1, post_fusion_context_vector.shape[1], 1
                        # ),
                    ],
                    dim=-1,
                )

        if self.mlm:
            """
            Masked Language Model
            """
            label_seq_trg, label_mlm_mask = mask_tokens(
                label_seq,
                self.mlm_probability,
                self.pad_token,
                self.device,
                self.training,
            )

        else:
            """
            Causal Language Modeling - Predict Next token
            """

            label_seq_inp = label_seq[:, :-1]
            label_seq_trg = label_seq[:, 1:]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
            label_seq_inp = torch.cat(
                [
                    label_seq_inp,
                    torch.zeros(
                        (label_seq_inp.shape[0], 1), dtype=label_seq_inp.dtype
                    ).to(self.device),
                ],
                axis=-1,
            )
            label_seq_trg = torch.cat(
                [
                    label_seq_trg,
                    torch.zeros(
                        (label_seq_trg.shape[0], 1), dtype=label_seq_trg.dtype
                    ).to(self.device),
                ],
                axis=-1,
            )

            # apply mask on input where target is on padding token
            mask_trg_pad = label_seq_trg != self.pad_token

            label_seq_inp = label_seq_inp * mask_trg_pad

            # When evaluating, computes metrics only for the last item of the session
            if (self.eval_on_last_item_seq_only and not self.training) or (
                self.train_on_last_item_seq_only and self.training
            ):
                rows_ids = torch.arange(
                    label_seq_inp.size(0), dtype=torch.long, device=self.device
                )
                last_item_sessions = mask_trg_pad.sum(axis=1) - 1
                label_seq_trg_eval = torch.zeros(
                    label_seq_trg.shape, dtype=torch.long, device=self.device
                )
                label_seq_trg_eval[rows_ids, last_item_sessions] = label_seq_trg[
                    rows_ids, last_item_sessions
                ]
                # Updating labels and mask
                label_seq_trg = label_seq_trg_eval
                mask_trg_pad = label_seq_trg != self.pad_token

        # Creating an additional feature with the position in the sequence
        metadata_for_pred_logging["seq_pos"] = torch.arange(
            1, label_seq.shape[1] + 1, device=self.device
        ).repeat(label_seq.shape[0], 1)
        metadata_for_pred_logging["seq_len"] = (
            (label_seq != self.pad_token)
            .int()
            .sum(axis=1)
            .unsqueeze(-1)
            .repeat(1, label_seq.shape[1])
        )
        # Keeping only metadata features for the next-clicks (targets)
        if not (self.mlm and self.training):
            for feat_name in metadata_for_pred_logging:
                metadata_for_pred_logging[feat_name] = metadata_for_pred_logging[
                    feat_name
                ][:, 1:]

                # As after shifting the sequence length will be subtracted by one, adding a masked item in
                # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
                metadata_for_pred_logging[feat_name] = torch.cat(
                    [
                        metadata_for_pred_logging[feat_name],
                        torch.zeros(
                            (metadata_for_pred_logging[feat_name].shape[0], 1),
                            dtype=metadata_for_pred_logging[feat_name].dtype,
                        ).to(self.device),
                    ],
                    axis=-1,
                )

        # Step 2. Merge features
        pos_emb = self.merge(pos_inp)

        if self.mlm:
            # Masking inputs (with trainable [mask] embedding]) at masked label positions
            pos_emb_inp = torch.where(
                label_mlm_mask.unsqueeze(-1).bool(),
                self.masked_item_embedding.to(pos_emb.dtype),
                pos_emb,
            )
        else:
            # Truncating the input sequences length to -1
            pos_emb_inp = pos_emb[:, :-1]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
            pos_emb_inp = torch.cat(
                [
                    pos_emb_inp,
                    torch.zeros(
                        (pos_emb_inp.shape[0], 1, pos_emb_inp.shape[2]),
                        dtype=pos_emb_inp.dtype,
                    ).to(self.device),
                ],
                axis=1,
            )

            # Replacing the inputs corresponding to masked label with a trainable embedding
            pos_emb_inp = torch.where(
                mask_trg_pad.unsqueeze(-1).bool(),
                pos_emb_inp,
                self.masked_item_embedding.to(pos_emb_inp.dtype),
            )

            # Context vector of the next item (label)
            post_fusion_context_vector = post_fusion_context_vector[:, 1:]

            # As after shifting the sequence length will be subtracted by one, adding a masked item in
            # the sequence to return to the initial sequence. This is important for ReformerModel(), for example
            post_fusion_context_vector = torch.cat(
                [
                    post_fusion_context_vector,
                    torch.zeros(
                        (
                            post_fusion_context_vector.shape[0],
                            1,
                            post_fusion_context_vector.shape[2],
                        ),
                        dtype=pos_emb_inp.dtype,
                    ).to(self.device),
                ],
                axis=1,
            )

        # Step3. Run forward pass on model architecture

        if not isinstance(self.model, PreTrainedModel):  # Checks if its a transformer
            # compute output through RNNs
            results = self.model(input=pos_emb_inp)

            if type(results) is tuple or type(results) is list:
                pos_emb_pred = results[0]
            else:
                pos_emb_pred = results

            model_outputs = (None,)

        else:
            """
            Transformer Models
            """

            model_outputs = self.model(inputs_embeds=pos_emb_inp)

            pos_emb_pred = model_outputs[0]
            model_outputs = tuple(model_outputs[1:])

        if self.modules_merge == "elementwise":
            # Late join on contextual features
            pos_emb_pred = torch.multiply(
                pos_emb_pred, 1.0 + post_fusion_context_vector
            )
        elif self.modules_merge == "elementwise_sum":
            pos_emb_pred = pos_emb_pred + post_fusion_context_vector
        elif self.modules_merge == "concat":
            pos_emb_pred = torch.cat(
                [pos_emb_pred, post_fusion_context_vector], axis=-1
            )
            # logger.info("pos_emb_pred.shape = {}".format(pos_emb_pred.shape))
            pos_emb_pred = self.modules_merge_fc(pos_emb_pred)
        elif self.modules_merge == "elementwise_context_concat_search":
            pos_emb_pred = torch.multiply(
                pos_emb_pred, 1.0 + post_fusion_context_vector
            )
            pos_emb_pred = torch.cat([pos_emb_pred, search_outputs_concat], axis=-1)
            pos_emb_pred = self.modules_merge_fc(pos_emb_pred)

        pos_emb_pred = self.tf_out_act(self.transformer_output_project(pos_emb_pred))

        trg_flat = label_seq_trg.flatten()
        non_pad_mask = trg_flat != self.pad_token

        labels_all = torch.masked_select(trg_flat, non_pad_mask)

        # Step4. Compute logit and label for neg+pos samples

        # remove zero padding elements
        pos_emb_pred = self.remove_pad_3d(pos_emb_pred, non_pad_mask)

        if not self.mlm:
            # Keeping removing zero-padded items metadata features for the next-clicks (targets), so that they are aligned
            for feat_name in metadata_for_pred_logging:
                metadata_for_pred_logging[feat_name] = torch.masked_select(
                    metadata_for_pred_logging[feat_name].flatten(), non_pad_mask
                )

        if self.mf_constrained_embeddings:

            logits_all = F.linear(
                pos_emb_pred,
                weight=self.embedding_tables[self.label_embedding_table_name].weight,
                bias=self.output_layer_bias,
            )
        else:
            logits_all = self.output_layer(pos_emb_pred)

        # Softmax temperature to reduce model overconfidence and better calibrate probs and accuracy
        logits_all = torch.div(logits_all, self.softmax_temperature)

        predictions_all = self.log_softmax(logits_all)
        loss_ce = self.loss_nll(predictions_all, labels_all)
        loss = loss_ce
        # accuracy
        # _, max_idx = torch.max(logits_all, dim=1)
        # train_acc = (max_idx == labels_all).mean(dtype = torch.float32)
        # Scaling the loss
        loss = loss * self.loss_scale_factor

        outputs = {
            "loss": loss,
            "labels": labels_all,
            "predictions": logits_all,
            "pred_metadata": metadata_for_pred_logging,
            "model_outputs": model_outputs,  # Keep mems, hidden states, attentions if there are in it
        }

        return outputs

    def define_features_layers(self, model_args):
        self.target_dim = None
        self.label_feature_name = None
        self.label_embedding_table_name = None
        self.label_embedding_dim = None

        self.embedding_tables = nn.ModuleDict()
        self.features_embedding_projection_to_item_embedding_dim_layers = (
            nn.ModuleDict()
        )

        self.features_layer_norm = nn.ModuleDict()

        self.input_combined_dim = 0

        # set embedding tables
        for cname, cinfo in self.feature_map.items():
            if (
                "ignore_on_main_seq_feature_processing" in cinfo
                and cinfo["ignore_on_main_seq_feature_processing"]
            ):
                continue

            if cinfo["dtype"] == "categorical":
                if self.use_ohe_item_ids_inputs:
                    feature_size = cinfo["cardinality"]
                else:
                    if "is_label" in cinfo and cinfo["is_label"]:
                        if model_args.item_embedding_dim is not None:
                            embedding_size = model_args.item_embedding_dim
                        # This condition is just to keep compatibility with the experiments of SIGIR paper
                        # (where item embedding dim was always equal to d_model)
                        elif model_args.mf_constrained_embeddings:
                            embedding_size = model_args.d_model
                        else:
                            embedding_size = get_embedding_size_from_cardinality(
                                cinfo["cardinality"],
                                multiplier=self.embedding_dim_from_cardinality_multiplier,
                            )
                        feature_size = embedding_size

                        self.item_embedding_dim = embedding_size

                        self.label_feature_name = cname
                        self.label_embedding_table_name = cinfo["emb_table"]
                        self.label_embedding_dim = embedding_size

                    else:
                        if self.features_same_size_item_embedding:
                            if self.label_embedding_dim:
                                embedding_size = self.label_embedding_dim
                                feature_size = embedding_size
                            else:
                                raise ValueError(
                                    "Make sure that the item id (label feature) is the first in the YAML features config file."
                                )
                        else:
                            embedding_size = get_embedding_size_from_cardinality(
                                cinfo["cardinality"],
                                multiplier=self.embedding_dim_from_cardinality_multiplier,
                            )
                            feature_size = embedding_size

                    self.embedding_tables[cinfo["emb_table"]] = nn.Embedding(
                        cinfo["cardinality"],
                        embedding_size,
                        padding_idx=self.pad_token,
                    ).to(self.device)

                    # Added to initialize embeddings
                    if "is_label" in cinfo and cinfo["is_label"]:
                        embedding_init_std = self.item_id_embeddings_init_std
                    else:
                        embedding_init_std = self.other_embeddings_init_std

                    with torch.no_grad():
                        self.embedding_tables[cinfo["emb_table"]].weight.normal_(
                            0.0, embedding_init_std
                        )

                logger.info(
                    "Categ Feature: {} - Cardinality: {} - Feature Size: {}".format(
                        cname, cinfo["cardinality"], feature_size
                    )
                )

            elif cinfo["dtype"] in ["long", "float"]:

                feature_size = 1

                logger.info(
                    "Numerical Feature: {} - Feature Size: {}".format(
                        cname, feature_size
                    )
                )

            elif cinfo["is_control"]:
                # Control features are not used as input for the model
                continue
            else:
                raise NotImplementedError

            self.input_combined_dim += feature_size

            if self.layer_norm_featurewise:
                self.features_layer_norm[cname] = nn.LayerNorm(
                    normalized_shape=feature_size
                )

            if "is_label" in cinfo and cinfo["is_label"]:
                self.target_dim = cinfo["cardinality"]

        if self.target_dim == None:
            raise RuntimeError("label column is not declared in feature map.")

    def feature_process(self, inputs):

        label_seq, output = None, []
        metadata_for_pred_logging = {}

        transformed_features = OrderedDict()
        for cname, cinfo in self.feature_map.items():

            cdata = inputs[cname]

            if (
                "ignore_on_main_seq_feature_processing" in cinfo
                and cinfo["ignore_on_main_seq_feature_processing"]
            ):
                continue

            if "is_label" in cinfo and cinfo["is_label"]:
                label_seq = cdata

            if cinfo["dtype"] == "categorical":
                cdata = cdata.long()

                # Applies Stochastic Shared Embeddings if training
                if (
                    self.stochastic_shared_embeddings_replacement_prob > 0.0
                    and not self.use_ohe_item_ids_inputs
                    and self.training
                ):
                    with torch.no_grad():
                        cdata_non_zero_mask = cdata != self.pad_token

                        sse_prob_replacement_matrix = torch.full(
                            cdata.shape,
                            self.stochastic_shared_embeddings_replacement_prob,
                            device=self.device,
                        )
                        sse_replacement_mask = (
                            torch.bernoulli(sse_prob_replacement_matrix).bool()
                            & cdata_non_zero_mask
                        )
                        n_values_to_replace = sse_replacement_mask.sum()

                        cdata_flattened_non_zero = torch.masked_select(
                            cdata, cdata_non_zero_mask
                        )

                        sampled_values_to_replace = cdata_flattened_non_zero[
                            torch.randperm(cdata_flattened_non_zero.shape[0])
                        ][:n_values_to_replace]

                        cdata[sse_replacement_mask] = sampled_values_to_replace

                if "is_label" in cinfo and cinfo["is_label"]:
                    if self.use_ohe_item_ids_inputs:
                        cdata = torch.nn.functional.one_hot(
                            cdata, num_classes=self.target_dim
                        ).float()
                    else:
                        cdata = self.embedding_tables[cinfo["emb_table"]](cdata)
                else:
                    cdata = self.embedding_tables[cinfo["emb_table"]](cdata)

            elif cinfo["dtype"] in ["long", "float"]:
                if cinfo["dtype"] == "long":
                    cdata = cdata.unsqueeze(-1).long()
                elif cinfo["dtype"] == "float":
                    cdata = cdata.unsqueeze(-1).float()

            elif cinfo["is_control"]:
                # Control features are not used as input for the model
                continue
            else:
                raise NotImplementedError

            # Applying layer norm for each feature
            if self.layer_norm_featurewise:
                cdata = self.features_layer_norm[cname](cdata)

            transformed_features[cname] = cdata

            # Keeping item metadata features that will
            if (
                "log_with_preds_as_metadata" in cinfo
                and cinfo["log_with_preds_as_metadata"] == True
            ):
                metadata_for_pred_logging[cname] = inputs[cname].detach()

        if len(transformed_features) > 1:
            if self.input_features_aggregation == "concat":
                output = torch.cat(list(transformed_features.values()), dim=-1)
            else:
                raise ValueError("Invalid value for --input_features_aggregation.")
        else:
            output = list(transformed_features.values())[0]

        return output, label_seq, metadata_for_pred_logging

    def define_product_description_image_embeddings(self):
        # Converting numpy tensors to torch
        for k, v in self.prod_embeddings_dict.items():
            self.prod_embeddings_dict[k] = torch.tensor(v, device=self.device)

        empty_descr_embeddings_mask = (
            self.prod_embeddings_dict["description_embedding_matrix"].max(axis=1).values
            == 0
        )
        empty_image_embeddings_mask = (
            self.prod_embeddings_dict["image_embedding_matrix"].max(axis=1).values == 0
        )

        descr_embeddings_mean = self.prod_embeddings_dict[
            "description_embedding_matrix"
        ][~empty_descr_embeddings_mask].mean()
        descr_embeddings_std = self.prod_embeddings_dict[
            "description_embedding_matrix"
        ][~empty_descr_embeddings_mask].std()
        image_embeddings_mean = self.prod_embeddings_dict["image_embedding_matrix"][
            ~empty_image_embeddings_mask
        ].mean()
        image_embeddings_std = self.prod_embeddings_dict["image_embedding_matrix"][
            ~empty_image_embeddings_mask
        ].std()

        # Creating an embedding for empty vectors
        self.empty_prod_description_embedding = torch.normal(
            mean=descr_embeddings_mean,
            std=descr_embeddings_std / 2,
            size=(PRODUCT_QUERY_EMBEDDINGS_SIZE,),
        )

        self.empty_prod_image_embedding = torch.normal(
            mean=image_embeddings_mean,
            std=image_embeddings_std / 2,
            size=(PRODUCT_QUERY_EMBEDDINGS_SIZE,),
        )

        self.prod_embeddings_dict["description_embedding_matrix"][
            empty_descr_embeddings_mask
        ] = self.empty_prod_description_embedding
        self.prod_embeddings_dict["image_embedding_matrix"][
            empty_image_embeddings_mask
        ] = self.empty_prod_image_embedding

    def remove_pad_3d(self, inp_tensor, non_pad_mask):
        # inp_tensor: (n_batch x seqlen x emb_dim)
        inp_tensor = inp_tensor.flatten(end_dim=1)
        inp_tensor_fl = torch.masked_select(
            inp_tensor, non_pad_mask.unsqueeze(1).expand_as(inp_tensor)
        )
        out_tensor = inp_tensor_fl.view(-1, inp_tensor.size(1))
        return out_tensor


def mask_tokens(itemid_seq, mlm_probability, pad_token, device, training):
    """
    prepare sequence with mask for masked language modeling prediction
    the function is based on HuggingFace's transformers/data/data_collator.py

    INPUT:
    itemid_seq: sequence of input itemid (label) column
    mlm_probability: probability of an item to be selected (masked) to be a label for this sequence. P.s. We enforce that at least one item is masked for each sequence, so that the network can learn something with it.

    OUTPUT:
    labels: item id sequence as label
    masked_labels: bool mask with is true only for masked labels (targets)
    """

    # labels = itemid_seq.clone()
    labels = torch.full(
        itemid_seq.shape, pad_token, dtype=itemid_seq.dtype, device=device
    )
    non_padded_mask = itemid_seq != pad_token

    rows_ids = torch.arange(itemid_seq.size(0), dtype=torch.long, device=device)
    # During training, masks labels to be predicted according to a probability, ensuring that each session has at least one label to predict
    if training:
        # Selects a percentage of items to be masked (selected as labels)
        probability_matrix = torch.full(
            itemid_seq.shape, mlm_probability, device=device
        )
        masked_labels = torch.bernoulli(probability_matrix).bool() & non_padded_mask
        labels = torch.where(
            masked_labels, itemid_seq, torch.full_like(itemid_seq, pad_token),
        )

        # Set at least one item in the sequence to mask, so that the network can learn something with this session
        one_random_index_by_session = torch.multinomial(
            non_padded_mask.float(), num_samples=1
        ).squeeze()
        labels[rows_ids, one_random_index_by_session] = itemid_seq[
            rows_ids, one_random_index_by_session
        ]
        masked_labels = labels != pad_token

        # If a sequence has only masked labels, unmasks one of the labels
        sequences_with_only_labels = masked_labels.sum(axis=1) == non_padded_mask.sum(
            axis=1
        )
        sampled_labels_to_unmask = torch.multinomial(
            masked_labels.float(), num_samples=1
        ).squeeze()

        labels_to_unmask = torch.masked_select(
            sampled_labels_to_unmask, sequences_with_only_labels
        )
        rows_to_unmask = torch.masked_select(rows_ids, sequences_with_only_labels)

        labels[rows_to_unmask, labels_to_unmask] = pad_token
        masked_labels = labels != pad_token

        # Logging the real percentage of masked items (labels)
        # perc_masked_labels = masked_labels.sum() / non_padded_mask.sum().float()
        # logger.info(f"  % Masked items as labels: {perc_masked_labels}")

    # During evaluation always masks the last item of the session
    else:
        last_item_sessions = non_padded_mask.sum(axis=1) - 1
        labels[rows_ids, last_item_sessions] = itemid_seq[rows_ids, last_item_sessions]
        masked_labels = labels != pad_token

    return labels, masked_labels


# From https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch
class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        # The following line was commented because the inpus were already processed by log_softmax()
        # lsm = F.log_softmax(inputs, -1)
        lsm = inputs

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss
