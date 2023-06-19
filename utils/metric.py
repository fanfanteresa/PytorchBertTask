# -*- coding: utf-8 -*-
import logging
from typing import List
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

logger = logging.getLogger()


def calculate_metric(logits: List[torch.Tensor], labels: List[torch.Tensor]):
    y_pred = []
    y_true = []
    for p, l in zip(logits, labels):
        p = p.cpu()
        l = l.cpu()
        values, indices = p.max(-1)
        y_pred.extend(indices.numpy().tolist())
        y_true.extend(l.numpy().tolist())

    logger.info(" ")
    logger.info("=" * 20)
    logger.info(classification_report(y_true, y_pred))
    logger.info("=" * 20)
    logger.info(" ")

    res = classification_report(y_true, y_pred, output_dict=True)['weighted avg']

    return res["precision"], res["recall"], res["f1-score"]



