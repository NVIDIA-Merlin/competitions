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


'''
this code was adapted from this example nb:
https://github.com/NVIDIA/NVTabular/blob/main/examples/tensorflow/accelerating-tensorflow.ipynb
'''

'''
The example script to run this file:

python tfdlrm-sigir.py --output_dir tmp_dlrm --data_path /workspace/SIGIR-ecom-data-challenge/data/ --lr 0.001 --feature_config_cat cat_features_dlrm.txt --feature_config_num num_features_dlrm.txt --emb_dim 64 --fold 1 --bag_number 4 --epochs 3 
'''


import numpy as np
import pandas as pd
import tensorflow as tf
import random 
import os
from itertools import filterfalse
import re
import argparse
import datetime
import glob
import json
import cudf
from time import time


from evaluation import cart_abandonment_metric

# TensorFlow's default behavior is to claim all of the GPU memory that it can for itself. 
# This is a problem when it needs to run alongside another GPU library
# like NVTabular. To get around this, NVTabular will configure
# TensorFlow to use this fraction of available GPU memory up front.
# Make sure, however, that you do this before you do anything
# with TensorFlow: as soon as it's initialized, that memory is gone for good

os.environ["TF_MEMORY_ALLOCATION"] = "0.7"
import nvtabular as nvt
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.framework_utils.tensorflow import layers, make_feature_column_workflow
from tensorflow.keras import regularizers

from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger

# define high cardinality categorical for Target Encoding
high_card_cats = ['product_url_id_list_after-0', 'product_url_id_list_after-1', 'product_url_id_list_after-2',
        'product_url_id_list_after-3', 'product_url_id_list_after-4', 'product_url_id_list_before-0', 
        'product_url_id_list_before-1', 'product_url_id_list_before-2', 'product_url_id_list_before-3', 
        'product_url_id_list_before-4',
        'event_type_list_after-0', 'event_type_list_after-1', 'event_type_list_after-2',
        'product_action_list_after-0', 'product_action_list_before-0',
        'category_list_before-0', 'category_list_before-1', 'category_list_before-2',
        'category_list_after-0', 'category_list_after-1', 'category_list_after-2']

def generate_card(df, cat_names):
    cardinalities = {}
    for col in cat_names:
        cardinalities[col]=df[col].nunique()
    return cardinalities
                              
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed+1)
    tf.random.set_seed(seed)
    
def load_fold_data(train_df_files, valid_df_file, test_df, TE_COLS=high_card_cats, data_path=None): 
    
    train_frames = pd.concat([pd.read_parquet(file) for file in train_df_files])
    valid_frames = cudf.read_parquet(valid_df_file)
    test_frames = test_df
   
    if len(TE_COLS) > 0:
        print("target encoding..")
        other_cols = [col for col in train_frames.columns if col not in TE_COLS] 
        te_features = TE_COLS >> nvt.ops.TargetEncoding("is_purchased-last", p_smooth=20)
        df_train = nvt.Dataset(train_frames)
        workflow = nvt.Workflow(other_cols + te_features)
        # fit Target Encoding model
        workflow.fit(df_train)
        # transform features 
        train_frames = workflow.transform(nvt.Dataset(train_frames)).to_ddf().compute().to_pandas()
        valid_frames = workflow.transform(nvt.Dataset(valid_frames)).to_ddf().compute().to_pandas()
        test_frames = workflow.transform(nvt.Dataset(test_frames)).to_ddf().compute().to_pandas()

    # session id info 
    eval_information = valid_frames[['original_session_id_hash',  'nb_after_add-last']]
    test_information = test_frames[['original_session_id_hash',  'nb_after_add-last']]
    train_information = train_frames[['original_session_id_hash',  'nb_after_add-last']]
    return (train_frames, train_information), (valid_frames, eval_information), (test_frames, test_information)

    
def main(args):
    DLLOGGER_FILENAME = 'log.json'
    DLLogger.init(backends=[
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, os.path.join(args.log_dir, DLLOGGER_FILENAME)),
    ])

    # load feature names
    with open(args.feature_config_cat) as f:
        CATEGORICAL_FEATURE_NAMES = f.read().splitlines() 
        print("use DLRM with cat features: {}".format(CATEGORICAL_FEATURE_NAMES), "\n")
    
      # load feature names
    with open(args.feature_config_num) as f:
        NUMERIC_FEATURE_NAMES = f.read().splitlines() 
        print("use DLRM with num features: {}".format(NUMERIC_FEATURE_NAMES), "\n")
    
    LABEL_NAME = 'is_purchased-last'
    
    fold = args.fold
    bag_number = args.bag_number
    seed = (bag_number * 10) 
    set_seeds(seed)
    
    DATA_DIR = args.data_path
    log_dir = args.log_dir
    LOG_DIR = os.environ.get("LOG_DIR", log_dir)
    
    output_bag_folder = os.path.join(args.output_dir, f"bag_{bag_number}")
  
    sigir_metrics, aucs, cv_preds = [], [], []
    
    #num rows of training set
    num_rows = 36672493
    hashing = False
    offline_hashing = False
    hp= 1.0

    
    TRAIN_FULL_PARQUET = cudf.read_parquet(os.path.join(DATA_DIR, f"train-full.parquet"))
    VALID_FULL_PARQUET = cudf.read_parquet(os.path.join(DATA_DIR, f"valid-full.parquet"))
    TEST_FULL_PARQUET = cudf.read_parquet(os.path.join(DATA_DIR, f"test-full.parquet"))

    TRAIN_FULL_PARQUET[CATEGORICAL_FEATURE_NAMES] = TRAIN_FULL_PARQUET[CATEGORICAL_FEATURE_NAMES].astype('int64')
    VALID_FULL_PARQUET[CATEGORICAL_FEATURE_NAMES] = VALID_FULL_PARQUET[CATEGORICAL_FEATURE_NAMES].astype('int64')
    TEST_FULL_PARQUET[CATEGORICAL_FEATURE_NAMES] = TEST_FULL_PARQUET[CATEGORICAL_FEATURE_NAMES].astype('int64')
    
    VALID_PARQUET_FILE = os.path.join(DATA_DIR, f"valid-{fold}.parquet")
    
    training_datasets = ["train"]
    training_files = []
    for f in range(1, 6):
        if f != fold:
            for dataset in training_datasets:
                training_files.append(os.path.join(DATA_DIR, f"{dataset}-{f}.parquet"))

    TRAINING_PARQUET_FILES = training_files
    np.random.shuffle(TRAINING_PARQUET_FILES)

    print(f"Train parquet files: {TRAINING_PARQUET_FILES}")
    print(f"Valid parquet file: {VALID_PARQUET_FILE}")

    # load data 
    (X_train, eval_info_train), (X_validation, validation_predictions_df), (X_test, test_predictions_df) = load_fold_data(TRAINING_PARQUET_FILES, VALID_PARQUET_FILE, TEST_FULL_PARQUET, data_path=DATA_DIR)
    
    df_all = cudf.concat([TRAIN_FULL_PARQUET, VALID_FULL_PARQUET, TEST_FULL_PARQUET], ignore_index=True)
    # define categ and cont features.
    cardinality_train = generate_card(df_all, CATEGORICAL_FEATURE_NAMES)

    CATEGORY_COUNTS = []
    for col in CATEGORICAL_FEATURE_NAMES:
        CATEGORY_COUNTS.append(cardinality_train[col])
    print("CATEGORY_COUNTS:", CATEGORY_COUNTS)
      
    # optimization params
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    
    EMBEDDING_DROPOUT_RATE = args.emb_dropout_rate
    DROPOUT_RATE = args.dropout_rate
    # architecture params
    EMBEDDING_DIM = args.emb_dim
    TOP_MLP_HIDDEN_DIMS = [1024, 512, 256]
    BOTTOM_MLP_HIDDEN_DIMS = [1024, 1024, 512, 256]
    th = args.threshold
    tf.get_logger().setLevel('ERROR')
    
    
    def get_feature_columns(hash=False):
        columns = [tf.feature_column.numeric_column(name, (1,)) for name in NUMERIC_FEATURE_NAMES]
        for feature_name, count in zip(CATEGORICAL_FEATURE_NAMES, CATEGORY_COUNTS):
            bucketsize=count
            if hash:
                if bucketsize > int(2e+7):
                    bucketsize = int(hp * count)
                categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                    feature_name, bucketsize, dtype=tf.int64)
            else:
                categorical_column = tf.feature_column.categorical_column_with_identity(
                feature_name, bucketsize)
            embedding_column = tf.feature_column.embedding_column(categorical_column, EMBEDDING_DIM)
            columns.append(embedding_column)
        return columns

    def make_dataset(file, columns, train=True, hash=False):
        # make an nvtabular KerasSequenceLoader
        if train:
            dataset = KerasSequenceLoader(
                nvt.Dataset(file),
                batch_size=BATCH_SIZE,
                label_names=[LABEL_NAME],
                feature_columns=columns,
                shuffle=True,
                buffer_size=0.06,
                parts_per_chunk=1
            )
        else:
            dataset = KerasSequenceLoader(
                nvt.Dataset(file), 
                batch_size=BATCH_SIZE,
                label_names=[LABEL_NAME],
                feature_columns=columns,
                shuffle=False,
                buffer_size=0.06,
                parts_per_chunk=1
            )
        if hash:
            workflow, columns = make_feature_column_workflow(columns, LABEL_NAME)
            dataset.map(workflow)
        return dataset, columns

    
    class DLRMEmbedding(tf.keras.layers.Layer):
        def __init__(self, columns, **kwargs):
            is_cat = lambda col: hasattr(col, "categorical_column")
            embedding_columns = list(filter(is_cat, columns))
            numeric_columns = list(filterfalse(is_cat, columns))

            self.categorical_feature_names = [col.categorical_column.name for col in embedding_columns]
            self.numeric_feature_names = [col.name for col in numeric_columns]

            # we can do a much faster embedding that
            # doesn't break out the SparseTensor machinery
            self.categorical_densifier = layers.DenseFeatures(embedding_columns, aggregation="stack")
            self.categorical_reshape = None
            self.numeric_densifier = layers.DenseFeatures(numeric_columns, aggregation="concat")
            super(DLRMEmbedding, self).__init__(**kwargs)

        def call(self, inputs):
            if not isinstance(inputs, dict):
                raise TypeError("Expected a dict!")

            categorical_inputs = {name: inputs[name] for name in self.categorical_feature_names}
            numeric_inputs = {name: inputs[name] for name in self.numeric_feature_names}

            fm_x = self.categorical_densifier(categorical_inputs)
            fm_x = tf.keras.layers.Dropout(EMBEDDING_DROPOUT_RATE)(fm_x) #added by Ronay
            dense_x = self.numeric_densifier(numeric_inputs)
            if self.categorical_reshape is not None:
                fm_x = self.categorical_reshape(fm_x)
            return fm_x, dense_x

        def get_config(self):
            return {}

    
    class ReLUMLP(tf.keras.layers.Layer):
        def __init__(self, dims, output_activation, **kwargs):
            self.layers = []
            for dim in dims[:-1]:
                self.layers.append(tf.keras.layers.Dense(dim, activation="relu", 
                                                         kernel_regularizer=tf.keras.regularizers.L1L2(l1=args.l1, l2=args.l2)))
            self.layers.append(tf.keras.layers.Dense(dims[-1], activation=output_activation))
            super(ReLUMLP, self).__init__(**kwargs)

        def call(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def get_config(self):
            return {
                "dims": [layer.units for layer in self.layers],
                "output_activation": self.layers[-1].activation
            }
        
        
    class DLRM(tf.keras.layers.Layer):
        def __init__(self, embedding_dim, top_mlp_hidden_dims, bottom_mlp_hidden_dims, **kwargs):
            self.top_mlp = ReLUMLP(top_mlp_hidden_dims + [embedding_dim], "linear", name="top_mlp")
            self.bottom_mlp = ReLUMLP(bottom_mlp_hidden_dims + [1], "linear", name="bottom_mlp")
            self.interaction = layers.DotProductInteraction()

            # adding in an activation layer for stability for mixed precision training
            # not strictly necessary, but worth pointing out
            self.activation = tf.keras.layers.Activation("sigmoid", dtype="float32")
            self.double_check = tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(x, 0., 1.), dtype="float32")
            super(DLRM, self).__init__(**kwargs)

        def call(self, inputs):
            dense_x, fm_x = inputs
            dense_x = self.top_mlp(dense_x)
            dense_x_expanded = tf.expand_dims(dense_x, axis=1)

            x = tf.concat([fm_x, dense_x_expanded], axis=1)
            x = self.interaction(x)
            x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
            x = tf.concat([x, dense_x], axis=1)
            x = self.bottom_mlp(x)

            # stuff for mixed precision stability
            # not actually related to DLRM at all
            x = self.activation(x)
            x = self.double_check(x)
            return x

        def get_config(self):
            return {
                "embedding_dim": self.top_mlp.layers[-1].units,
                "top_mlp_hidden_dims": [layer.units for layer in self.top_mlp.layers[:-1]],
                "bottom_mlp_hidden_dims": [layer.units for layer in self.bottom_mlp.layers[:-1]]
            }

    # get our columns to describe our dataset
    columns = get_feature_columns(hash=hashing)

    # build a dataset from those descriptions
    train_dataset, columns = make_dataset(X_train, columns)
    valid_dataset, columns = make_dataset(X_validation, columns, train=False, hash=hashing)

    # build our Keras model, using column descriptions to build input tensors
    inputs = {}
    for column in columns:
        column = getattr(column, "categorical_column", column)
        dtype = getattr(column, "dtype", tf.int64)
        input = tf.keras.Input(name=column.name, shape=(1,), dtype=dtype)
        inputs[column.name] = input

    fm_x, dense_x = DLRMEmbedding(columns)(inputs)
    x = DLRM(EMBEDDING_DIM, TOP_MLP_HIDDEN_DIMS, BOTTOM_MLP_HIDDEN_DIMS)([dense_x, fm_x])
    model = tf.keras.Model(inputs=list(inputs.values()), outputs=x)

    # # compile our Keras model with our desired loss, optimizer, and metrics
    if args.optimizer == 'adam':
        print("using adam")
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    
    else:
        print("using sgd")
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=LEARNING_RATE,
            decay_steps=int(num_rows / BATCH_SIZE),
            end_learning_rate=LEARNING_RATE * 0.1,
            power=args.power)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=args.momentum, nesterov=bool(args.nesterov))

    #prepare the metrics
    auc_metric = tf.keras.metrics.AUC(curve="ROC", name="auc")
    acc_metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=th)

    val_auc_metric = tf.keras.metrics.AUC(curve="ROC", name="auc")
    val_acc_metric = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=th)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='binary_crossentropy')
    
    # Set up summary writers for TensorBoard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = '/result/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    start = time()
   
    #=============================
    #=============================
    # Training and validation loop
    for epoch in range(args.epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            auc_metric.update_state(y_batch_train, logits)
            acc_metric.update_state(y_batch_train, logits)

            train_auc = auc_metric.result()
            train_acc = acc_metric.result()

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=step)
                tf.summary.histogram('loss_hist', loss_value, step=step)
                tf.summary.scalar('logits_mean', tf.reduce_mean(logits), step=step)
                tf.summary.histogram('logits_hist', logits, step=step)
                tf.summary.scalar('train_acc', train_acc, step=step)
                tf.summary.scalar('train_auc', train_auc, step=step)

            # Log every 10 batches.
            if step % 10 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))

        # Display metrics at the end of each epoch.
        train_auc = auc_metric.result()
        train_acc = acc_metric.result()
        print("Training auc and acc over epoch: %.4f, %.4f" % (float(train_auc), float(train_acc)))
        # Reset training metrics at the end of each epoch
        auc_metric.reset_states()
        acc_metric.reset_states()
        training_time = time() - start
        print("total training time:", training_time)
        
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in valid_dataset:
            val_logits = model(x_batch_val, training=False)
            # Compute the loss value for this minibatch.
            val_loss_value = loss_fn(y_batch_val, val_logits)
            
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
            val_auc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_auc = val_auc_metric.result()

        val_acc_metric.reset_states()
        val_auc_metric.reset_states()
        print("Validation AUC and acc over epoch: %.4f, %.4f" % (float(val_auc), float(val_acc)))
    
    metrics = [tf.keras.metrics.AUC(curve="ROC", name="auc")]
    model.compile(optimizer, "binary_crossentropy", metrics=metrics)

    preds_val= model.predict(valid_dataset)
    data_val = pd.read_parquet(VALID_PARQUET_FILE)
    data_val['predictions'] = np.squeeze(preds_val)

    cart_abandonment_metric((data_val['predictions'].values > th).reshape(-1).astype(int).tolist(),
                        data_val['is_purchased-last'].values,
                        data_val['nb_after_add-last'])

    print("generating predictions for test set")
    test_dataset, columns = make_dataset(X_test, columns, train=False, hash=hashing)
    
    preds = model.predict(test_dataset)
    data = TEST_FULL_PARQUET
    data['predictions'] = np.squeeze(preds)
    
    output_fold_folder = os.path.join(output_bag_folder, f"fold_{fold}")
    os.makedirs(output_fold_folder, exist_ok=True)
    
    if args.save_pred:
        data[['original_session_id_hash', 'predictions', 'nb_after_add-last']].to_parquet(os.path.join(output_fold_folder, 'test_predictions_dlrm.parquet'))

        data_val[['original_session_id_hash', 'predictions','is_purchased-last', 
                  'nb_after_add-last']].to_parquet(os.path.join(output_fold_folder,'valid_predictions_dlrm.parquet'))
    
    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description=("Sigir Coveo Data Challenge"))
    parser.add_argument("--data_path", default='/workspace/SIGIR-ecom-data-challenge/xgboost_data_bal/', type=str, help="Input dataset path (Required)")
    parser.add_argument("--log_dir", default='/result/', type=str, help="Directory path to write logs (Required)")
    parser.add_argument('--output_dir',  type=str, 
                        help='output folder where results are saved')
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate ")
    parser.add_argument("--emb_dropout_rate", default=0.01, type=float, help="Embedding layer dropout rate")
    parser.add_argument("--dropout_rate", default=0.01, type=float, help="Hidden layer dropout rate")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size")
    parser.add_argument("--emb_dim", default=32, type=int, help="Embedding table dimension")
    parser.add_argument("--l1", default=1e-3, type=float, help="L1 regularization factor")
    parser.add_argument("--l2", default=1e-3, type=float, help="L2 regularization factor")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument("--momentum", default=0.5, type=float, help="sgd optimizer momentum param")
    parser.add_argument("--nesterov", default=True, type=str, help="sgd optimizer nesterov param")
    parser.add_argument("--power", default=1, type=float, help="PolynomialDecay param")
    parser.add_argument('--bag_number',  type=int,  default=1,
                        help='Number of training bags (each bag will use different seeds')
    parser.add_argument('--num_folds',  type=int, default=5,
                        help="Number of folds for each bag")
    parser.add_argument('--fold',  type=int, default=1,
                        help="the fold for each bag")
    parser.add_argument("--threshold", default=0.5, type=float, help="cut off for class probas")
    parser.add_argument('--feature_config_cat',  type=str, 
                        help='path to text file specifying the name of columns to use')
    parser.add_argument('--feature_config_num',  type=str, 
                        help='path to text file specifying the name of columns to use')
    parser.add_argument('--save_pred', action='store_true', 
                        help='use to save the prediction results in the bag folder')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)