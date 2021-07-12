# This code comes from the SIGIR eCommerce Data Challenge competition repo ( https://github.com/coveooss/SIGIR-ecom-data-challenge ) as an example to submit the prediction files for scoring in the Leaderboard.
"""
    In this script, we demonstrate how we perform the quantitative evaluation of the submissions
    for our two tasks, rec and cart.

    For information about the general evaluation procedure and relevant context, please refer to the README
    and the Data Challenge paper.
"""
import json
import random
from collections import defaultdict
from random import randint

import numpy as np


def convert_list_to_top_K(items_list: list, topK: int):
    """
    Extract top_K items
    It is assumed that the list of items are sorted in descending order of importance/weight/relevance

    :param items_list: list where element is a list of items
    :param topK: top K limit
    :return: list of items shortened with top K
    """
    return [list(items[:topK]) for items in items_list]


def mrr_at_k(preds: list, labels: list, topK: int):
    assert len(labels) > 0
    assert len(preds) == len(labels)

    # get top K predictions
    converted_preds = convert_list_to_top_K(items_list=preds, topK=topK)
    rr = []
    for p, l in zip(converted_preds, labels):
        if len(l) == 0:
            rr.append(0.0)
        else:
            # get next_item from labels
            next_item = l[0]
            # add 0.0 explicitly if not there (for transparency)
            if next_item not in p:
                rr.append(0.0)
            # else, take the reciprocal of prediction rank
            else:
                rr.append(1.0 / (p.index(next_item) + 1))

    # return the mean reciprocal rank
    return sum(rr) / len(labels)


def f1_at_k(preds: list, labels: list, topK: int):
    assert len(labels) > 0
    assert len(preds) == len(labels)

    # get top K predictions
    converted_preds = convert_list_to_top_K(items_list=preds, topK=topK)
    all_precision = []
    all_recall = []
    all_f1 = []

    # for each recommendation prediction, calculate the f1 based on ground truth
    for p, l in zip(converted_preds, labels):
        nb_hits = len(set(p).intersection(set(l)))
        precision = nb_hits / topK
        recall = nb_hits / len(set(l)) if len(l) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    # get average f1 across all predictions
    avg_f1 = sum(all_f1) / len(labels)
    return avg_f1


def weighted_micro_f1(preds, labels, nb_after_add, weights: dict):
    assert len(labels) > 0
    assert len(preds) == len(labels)
    assert len(labels) == len(nb_after_add)
    assert all(_ in [0, 2, 4, 6, 8, 10] for _ in nb_after_add)

    nb_added_2_preds_and_labels = defaultdict(list)
    for p, l, n in zip(preds, labels, nb_after_add):
        nb_added_2_preds_and_labels[n].append({"pred": p, "label": l})

    metric_to_score = {}
    for n, p_and_l in nb_added_2_preds_and_labels.items():
        p = [_["pred"] for _ in p_and_l]
        l = [_["label"] for _ in p_and_l]
        assert len(p) == len(l)
        num_correct = sum([1 for y, y_hat in zip(l, p) if y == y_hat])
        micro_f1 = num_correct / len(p_and_l)
        metric_to_score[n] = micro_f1
    weighted_sum = sum([f1 * weights[n] for n, f1 in metric_to_score.items()])

    return weighted_sum


def next_item_metric(preds: list, labels: list, top_K: int = 20):
    """
    Compute metric for next item recommendation task.

    :param preds: list where each element is a list of item recommendations
    :param labels: list where each element is a list of ground truth items viewed
    :param top_K: as per README, by default we consider metric @ K=20
    :return: mrr for next item prediction
    """
    return mrr_at_k(preds, labels, top_K)


def subsequent_items_metric(preds: list, labels: list, top_K: int = 20):
    """
    Compute metric for the all items recommendation task.

    :param preds:
    :param labels:
    :param top_K: as per README, by default we consider metric @ K=20
    :return: f1 for all items prediction
    """
    return f1_at_k(preds, labels, top_K)


def cart_abandonment_metric(preds: list, labels: list, nb_after_add: list):
    """
    Compute metric for the cart abandonment task.

    :param preds: list where each element is a cart-abandonment prediction
    :param labels: list where each element is cart-abandonment ground truth label
    :param nb_after_add: list where each element denotes number of events after the add-to-cart event
    :return: weighted micro-f1 for cart abandonment preidciton
    """

    # weights for N events after add-to-cart (refer to README)
    weights = {0: 1.0, 2: 0.9, 4: 0.8, 6: 0.7, 8: 0.6, 10: 0.5}
    return weighted_micro_f1(preds, labels, nb_after_add, weights)


def evaluate_recs(path_to_predictions: str, path_to_ground_truth: str):
    """
    This function computes the leaderboard metrics for the in-session recommendation task

    :param path_to_predictions: path of json file which stores predictions
    :param path_to_ground_truth: path of json file which stores ground truth
    :return: dictionary storing metrics for session-based recommendation task
    """

    # read predictions file
    with open(path_to_predictions) as f:
        prediction_data = json.load(f)

    # read ground truth file
    with open(path_to_ground_truth) as f:
        ground_truth_data = json.load(f)

    # check for 'label' key in every prediction item and ground truth item
    assert all("label" in _ for _ in prediction_data)
    assert all("label" in _ for _ in ground_truth_data)

    # extract prediction and ground truth labels
    predictions = [_["label"] for _ in prediction_data]
    ground_truth = [_["label"] for _ in ground_truth_data]

    next_item_mrr = next_item_metric(predictions, ground_truth)
    subsequent_items_f1 = subsequent_items_metric(predictions, ground_truth)

    return {"mrr_next_item": next_item_mrr, "f1_all_items": subsequent_items_f1}


def evaluate_cart(path_to_predictions: str, path_to_ground_truth: str):
    """
    This function computes the leaderboard metrics for the cart-abandonmnet prediction task

    :param path_to_predictions: path of json file which stores predictions
    :param path_to_ground_truth: path of json file which stores ground truth
    :return: dictionary storing metrics for session-based recommendation task
    """

    # read predictions file
    with open(path_to_predictions) as f:
        prediction_data = json.load(f)

    # read ground truth file
    with open(path_to_ground_truth) as f:
        ground_truth_data = json.load(f)

    # check for 'label' key in every prediction item and ground truth item
    assert all("label" in _ for _ in prediction_data)
    assert all("label" in _ for _ in ground_truth_data)
    assert all("nb_after_add" in _ for _ in ground_truth_data)

    # extract prediction and groun truth labels
    predictions = [_["label"] for _ in prediction_data]
    ground_truth = [_["label"] for _ in ground_truth_data]

    # required for computing weighted f1
    nb_after_add = [_["nb_after_add"] for _ in ground_truth_data]
    weighted_f1 = cart_abandonment_metric(predictions, ground_truth, nb_after_add)

    return {"weighted_micro_f1": weighted_f1}


def example_in_session_recommedation():
    """
    Demonstrate usage of evaluation function for in-session recommendation on dummy data
    """
    random.seed(0)
    np.random.seed(0)

    # Generate some fake predictions and labels
    N = 1000  # number of fake data points
    num_rec = 50  # max number of recommendations per data point
    # assume we have 1K 'SKUS' where each SKU is an integer
    dummy_skus = np.arange(0, 1000)

    # generate dummy ground_truth for REC task by randomly sampling SKUs
    rec_gt_dummy = [
        {
            "label": np.random.choice(
                dummy_skus, size=randint(0, num_rec), replace=False
            ).tolist()
        }
        for i in range(N)
    ]
    # generate dummy predictions for REC task by random sampling SKUs
    rec_preds_dummy = [
        {
            "label": np.random.choice(
                dummy_skus, size=randint(0, num_rec), replace=False
            ).tolist()
        }
        for _ in rec_gt_dummy
    ]
    # save dummy data
    PATH_TO_GT_REC = "./dummy_gt_rec.json"
    PATH_TO_PRED_REC = "./dummy_pred_rec.json"

    with open(PATH_TO_GT_REC, "w") as f:
        json.dump(rec_gt_dummy, f)
    with open(PATH_TO_PRED_REC, "w") as f:
        json.dump(rec_preds_dummy, f)

    # call evaluation functions, passing paths to predictions and ground truth
    print(evaluate_recs(PATH_TO_PRED_REC, PATH_TO_GT_REC))

    return


def example_cart_abandonment():
    """
    Demonstrate usage of evaluation function for cart-abandonment prediction task on dummy data
    """
    random.seed(0)
    np.random.seed(0)

    # Generate some fake predictions and labels
    N = 1000  # number of fake data points
    num_after_add = np.arange(0, 12, 2).tolist()

    # generate dummy ground_truth for CART task by randomly sampling label and num_after_add
    cart_gt_dummy = [
        {"label": randint(0, 1), "nb_after_add": num_after_add[randint(0, 5)]}
        for i in range(N)
    ]

    # generate dummy predictions for CART task by random sampling label
    cart_preds_dummy = [{"label": randint(0, 1)} for _ in cart_gt_dummy]

    # save dummy data
    PATH_TO_GT_CART = "./dummy_gt_cart.json"
    PATH_TO_PRED_CART = "./dummy_pred_cart.json"

    with open(PATH_TO_GT_CART, "w") as f:
        json.dump(cart_gt_dummy, f)
    with open(PATH_TO_PRED_CART, "w") as f:
        json.dump(cart_preds_dummy, f)

    # call evaluation functions, passing paths to predictions and ground truth
    print(evaluate_cart(PATH_TO_PRED_CART, PATH_TO_GT_CART))

    return


if __name__ == "__main__":
    # if you run the script, the two mock functions will demonstrate the
    # evaluation for the two tasks in the Challenge
    example_in_session_recommedation()
    example_cart_abandonment()

