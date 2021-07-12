# This code comes from the SIGIR eCommerce Data Challenge competition repo ( https://github.com/coveooss/SIGIR-ecom-data-challenge ) as an example to submit the prediction files for scoring in the Leaderboard.
"""

    Sample script running e-2-e a prod2vec knn model for the rec task, including automatically submitting the
    local prediction file to AWS. This is just a sample script provided for your convenience, but it should not
    be treated as a credible baseline.

"""

import csv
import json
import os
import time
from datetime import datetime
from random import choice

import gensim
from dotenv import load_dotenv

from uploader import upload_submission

TRAIN_PATH = "/dataset/sigir_coveo_ecom_datachallenge/SIGIR-ecom-data-challenge/train/browsing_train.csv"
TEST_PATH = "/dataset/sigir_coveo_ecom_datachallenge/SIGIR-ecom-data-challenge/test/rec_test_phase_1.json"

# load envs from env file
load_dotenv(
    verbose=True,
    dotenv_path="sigir_ecom_challenge_code/submission/upload.env",
)


# read envs from file
EMAIL = os.getenv("EMAIL", None)  # the e-mail you used to sign up
assert EMAIL is not None


def train_product_2_vec_model(
    sessions: list,
    min_c: int = 3,
    size: int = 48,
    window: int = 5,
    iterations: int = 15,
    ns_exponent: float = 0.75,
):
    """
    Train CBOW to get product embeddings. We start with sensible defaults from the literature - please
    check https://arxiv.org/abs/2007.14906 for practical tips on how to optimize prod2vec.

    :param sessions: list of lists, as user sessions are list of interactions
    :param min_c: minimum frequency of an event for it to be calculated for product embeddings
    :param size: output dimension
    :param window: window parameter for gensim word2vec
    :param iterations: number of training iterations
    :param ns_exponent: ns_exponent parameter for gensim word2vec
    :return: trained product embedding model
    """
    model = gensim.models.Word2Vec(
        sentences=sessions,
        min_count=min_c,
        vector_size=size,
        window=window,
        epochs=iterations,
        ns_exponent=ns_exponent,
    )

    print("# products in the space: {}".format(len(model.wv.index_to_key)))

    return model.wv


def read_sessions_from_training_file(training_file: str, K: int = None):
    user_sessions = []
    current_session_id = None
    current_session = []
    with open(training_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            # if a max number of items is specified, just return at the K with what you have
            if K and idx >= K:
                break
            # just append "detail" events in the order we see them
            # row will contain: session_id_hash, product_action, product_sku_hash
            _session_id_hash = row["session_id_hash"]
            # when a new session begins, store the old one and start again
            if (
                current_session_id
                and current_session
                and _session_id_hash != current_session_id
            ):
                user_sessions.append(current_session)
                # reset session
                current_session = []
            # check for the right type and append
            if row["product_action"] == "detail":
                current_session.append(row["product_sku_hash"])
            # update the current session id
            current_session_id = _session_id_hash

    # print how many sessions we have...
    print("# total sessions: {}".format(len(user_sessions)))
    # print first one to check
    print("First session is: {}".format(user_sessions[0]))
    assert (
        user_sessions[0][0]
        == "d5157f8bc52965390fa21ad5842a8502bc3eb8b0930f3f8eafbc503f4012f69c"
    )
    assert (
        user_sessions[0][-1]
        == "63b567f4cef976d1411aecc4240984e46ebe8e08e327f2be786beb7ee83216d0"
    )

    return user_sessions


def make_predictions(prod2vec_model, test_file: str):
    cnt_preds = 0
    my_predictions = []
    # get all possible SKUs in the model, as a back-up choice
    all_skus = list(prod2vec_model.index_to_key)
    print("Same SKUS.. {}".format(all_skus[:2]))
    with open(test_file) as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)
    # loop over the records and predict the next event
    for t in test_queries:
        # this is our prediction, which defaults to a random SKU
        next_sku = choice(all_skus)
        # copy the test case
        _pred = dict(t)
        _products_in_session = [
            _["product_sku_hash"] for _ in t["query"] if _["product_sku_hash"]
        ]
        # get last product in the query session and check it is in the model space
        if _products_in_session and _products_in_session[-1] in all_skus:
            # get first product from knn
            next_sku = prod2vec_model.similar_by_word(_products_in_session[-1], topn=1)[
                0
            ][0]
            cnt_preds += 1

        # append the label - which needs to be a list
        _pred["label"] = [next_sku]
        # append prediction to the final list
        my_predictions.append(_pred)

    # check for consistency
    assert len(my_predictions) == len(test_queries)
    # print out some "coverage"
    print(
        "Predictions made in {} out of {} total test cases".format(
            cnt_preds, len(test_queries)
        )
    )

    return my_predictions


def train_knn(upload=False):
    """
    print("Starting training at {}...\n".format(datetime.utcnow()))
    #  read sessions in from browsing_train.csv in the folder
    #  to avoid waiting too much, we just train on the first K events
    sessions = read_sessions_from_training_file(training_file=TRAIN_PATH, K=300000)
    # train p2vec, leaving all the default hyperparameters
    prod2vec_model = train_product_2_vec_model(sessions)
    # use model to predict
    preds = make_predictions(prod2vec_model, test_file=TEST_PATH)
    """
    # name the prediction file according to the README specs
    # local_prediction_file = "{}_{}.json".format(
    #    EMAIL.replace("@", "_"), round(time.time() * 1000)
    # )
    local_prediction_file = "gmoreira_nvidia.com_1621793953608.json"
    """
    # dump to file
    with open(local_prediction_file, "w") as outfile:
        json.dump(preds, outfile, indent=2)
    # finally, upload the test file using the provided script
    """
    if upload:
        upload_submission(local_file=local_prediction_file, task="rec")
    # bye bye
    print("\nAll done at {}: see you, space cowboy!".format(datetime.utcnow()))

    return


if __name__ == "__main__":
    train_knn(upload=True)
