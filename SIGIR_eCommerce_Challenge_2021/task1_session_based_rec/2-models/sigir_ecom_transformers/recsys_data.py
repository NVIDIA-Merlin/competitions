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
Set data-specific schema, vocab sizes, and feature extract function.
"""

import logging
from random import randint


import torch

# NVTabular dependencies
from nvtabular import Dataset as NVTDataset
from nvtabular.loader.torch import DLDataLoader
from nvtabular.loader.torch import TorchAsyncItr as NVTDataLoader

logger = logging.getLogger(__name__)


class DLDataLoaderWrapper(DLDataLoader):
    def __init__(self, *args, **kwargs) -> None:
        if "batch_size" in kwargs:
            # Setting the batch size directly to DLDataLoader makes it 3x slower. So we set as an alternative attribute and use it within RecSysTrainer during evaluation
            self._batch_size = kwargs.pop("batch_size")
        super().__init__(*args, **kwargs)


def fetch_data_loader(
    data_args,
    training_args,
    feature_map,
    data_paths,
    is_train_set,
    shuffle_dataloader=False,
    add_dummy_item_end_sequence_label_column=False,
    label_column_name=None,
):

    if type(data_paths) is not list:
        data_paths = [data_paths]

    batch_size = (
        training_args.per_device_train_batch_size
        if is_train_set
        else training_args.per_device_eval_batch_size
    )

    loader = get_nvtabular_dataloader(
        data_args,
        training_args,
        feature_map,
        data_paths,
        batch_size,
        shuffle_dataloader,
        add_dummy_item_end_sequence_label_column,
        label_column_name=label_column_name,
    )

    return loader


def get_nvtabular_dataloader(
    data_args,
    training_args,
    feature_map,
    data_paths,
    batch_size,
    shuffle_dataloader=False,
    add_dummy_item_end_sequence_label_column=False,
    label_column_name=None,
):
    def dataloader_collate(inputs):
        # Gets only the features dict
        inputs = inputs[0][0]

        # Adds a dummy item in the end of item id sequence so that the last known item is not masked (only during inference)
        if add_dummy_item_end_sequence_label_column:
            sessions_length = torch.count_nonzero(inputs[label_column_name], dim=1)

            sessions_length = torch.min(
                sessions_length,
                torch.tensor(data_args.session_seq_length_max - 1),
                # torch.tensor(data_args.seq_features_len_pad_trim - 1),
            )
            rows_ids = torch.arange(sessions_length.size(0), dtype=torch.long)

            # Adding one additional item id in the sequence for inference
            inputs[label_column_name][rows_ids, sessions_length] = 1
            # Adding one additional item id in the sequence for inference
            EVENT_TYPE_PRODUCT = 1
            FREQ_ITEM_FLAG = 2  # Unfreq item flag is 1 and 0 is for padding
            # Set the contextual feature "event_type-list" so that the model can know that predictions should be on encoded product_skus and not hashed_urls (page views)
            inputs["event_type-list"][rows_ids, sessions_length] = EVENT_TYPE_PRODUCT
            # Set the contextual feature "product_url_hash_list_unfreq" so that the model can know that predictions should not be on unfrequent items

            if data_args.use_freq_cap_item_id:
                inputs["product_url_hash_list_unfreq"][
                    rows_ids, sessions_length
                ] = FREQ_ITEM_FLAG

        return inputs

    categ_features = []
    continuous_features = []
    for fname, fprops in feature_map.items():
        if fprops["dtype"] in ["categorical", "timestamp"]:
            categ_features.append(fname)
        elif fprops["dtype"] in ["float", "long"]:
            continuous_features.append(fname)
        else:
            raise NotImplementedError(
                "The dtype {} is not currently supported.".format(fprops["dtype"])
            )

    sparse_features_max = {
        fname: feature_map[fname]["pad_trim_length"]
        if fname in feature_map and "pad_trim_length" in feature_map[fname]
        else data_args.session_seq_length_max
        for fname in categ_features + continuous_features
    }

    dataloader_device = (
        0 if training_args.local_rank == -1 else training_args.local_rank
    )

    dataset = NVTDataset(
        data_paths,
        engine="parquet",
        part_mem_fraction=data_args.nvt_part_mem_fraction,
        part_size=data_args.nvt_part_size,
    )

    global_size = None
    global_rank = None
    # If using DistributedDataParallel, gets the global number of GPUs (world_size) and the GPU for this process (local_rank).
    # Each GPU will be assigned to one process and the data loader will read different chunks of data for each GPU
    if training_args.local_rank != -1:
        global_size = get_world_size()
        global_rank = training_args.local_rank

    loader = NVTDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataloader,
        global_size=global_size,
        global_rank=global_rank,
        device=dataloader_device,
        cats=categ_features,
        conts=continuous_features,
        labels=[],
        sparse_names=categ_features + continuous_features,
        sparse_max=sparse_features_max,
        sparse_as_dense=True,
        drop_last=training_args.dataloader_drop_last,
    )

    dl_loader = DLDataLoaderWrapper(
        loader, collate_fn=dataloader_collate, batch_size=batch_size
    )

    return dl_loader
