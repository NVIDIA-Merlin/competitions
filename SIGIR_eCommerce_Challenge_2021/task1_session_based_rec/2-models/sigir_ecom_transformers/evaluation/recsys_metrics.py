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
from typing import Dict, List

import numpy as np
import torch

from .ranking_metrics_torch.avg_precision import avg_precision_at
from .ranking_metrics_torch.cumulative_gain import ndcg_at
from .ranking_metrics_torch.precision_recall import precision_at, recall_at

METRICS_MAPPING = {
    "ndcg": ndcg_at,
    "map": avg_precision_at,
    "recall": recall_at,
    "precision": precision_at,
}


class EvalMetrics(object):
    def __init__(self, ks=[5, 10, 100, 1000]):
        self.ks = ks
        self.max_k = max(ks)

        self.f_measures_torch = []

        f_precision_kh = MetricWrapper("precision", precision_at, ks)
        f_recall_kh = MetricWrapper("recall", recall_at, ks)
        f_avgp_kh = MetricWrapper("avg_precision", avg_precision_at, ks)
        f_ndcg_kh = MetricWrapper("ndcg", ndcg_at, ks)
        self.f_measures_torch.extend(
            [f_precision_kh, f_recall_kh, f_avgp_kh, f_ndcg_kh]
        )

    def reset(self):
        for f_measure in self.f_measures_torch:
            f_measure.reset()

    def update(self, preds, labels, return_individual_metrics=False):
        metrics_results = {}

        # compute metrics on PyTorch
        labels = torch.nn.functional.one_hot(
            labels.reshape(-1), preds.size(-1)
        ).detach()
        preds = preds.view(-1, preds.size(-1))
        for f_measure in self.f_measures_torch:
            results = f_measure.add(
                preds, labels, return_individual_metrics=return_individual_metrics
            )
            # Merging metrics results
            if return_individual_metrics:
                metrics_results = {**metrics_results, **results}

        return metrics_results

    def result(self):
        metrics = []

        # PyTorch
        metrics.extend([f_measure.result() for f_measure in self.f_measures_torch])

        return {k: v for d in metrics for k, v in d.items()}


class MetricWrapper(object):
    def __init__(self, name, f_metric, topks):
        self.name = name
        self.topks = topks
        self.f_metric = f_metric
        self.reset()

    def reset(self):
        self.results = {k: [] for k in self.topks}

    def add(self, predictions, labels, return_individual_metrics=False):

        # represent target class id as one-hot vector
        # labels = torch.nn.functional.one_hot(labels, predictions.size(-1)).detach()

        # Computing the metrics at different cut-offs
        metric = self.f_metric(torch.LongTensor(self.topks), predictions, labels)

        # del(labels)

        # Retrieving individual metric results (for each next-item recommendation list), to return for debug logging purposes
        if return_individual_metrics:
            returns = {}
            for k, measures in zip(self.topks, metric.T):
                returns[f"{self.name}@{k}"] = measures.cpu().numpy()

        # Computing the mean of the batch metrics (for each cut-off at topk)
        metric_mean = metric.mean(0)

        # Storing in memory the average metric results for this batch
        for k, measure in zip(self.topks, metric_mean):
            self.results[k].append(measure.cpu().item())

        if return_individual_metrics:
            # Returning individual metric results, for debug logging purposes
            return returns

    def result(self):
        return {f"{self.name}@{k}": np.mean(self.results[k]) for k in self.topks}

