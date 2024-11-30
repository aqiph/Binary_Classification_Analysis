#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:48:00 2024

@author: aqiph

"""
import numpy as np
import scipy
from sklearn.metrics import precision_recall_curve, auc



### Precision and Recall ###
def calculate_PR(predicted_labels, true_labels):
    """
    Calculate precision and recall for single experiment.
    :param predicted_labels: list of ints, list of predicted labels, i.e., 1 or 0.
    :param true_labels: list of ints, list of true labels, i.e., 1, 0, Nan.
    :return: list of floats, precision_1, recall_1, precision_0, recall_0.
    """
    assert len(predicted_labels) == len(true_labels), 'Error: Number of predicted labels should be the same as that of true labels'
    TP, FP, TN, FN = 0, 0, 0, 0

    for i, pred_label in enumerate(predicted_labels):
        true_label = true_labels[i]
        try:
            pred_label, true_label = int(pred_label), int(true_label)
            if pred_label == 1 and true_label == 1:
                TP += 1
            elif pred_label == 1 and true_label == 0:
                FP += 1
            elif pred_label == 0 and true_label == 0:
                TN += 1
            elif pred_label == 0 and true_label == 1:
                FN += 1
        except:
            continue

    # precision for label 1
    precision_1 = np.nan if TP + FP == 0 else TP * 1.0 / (TP + FP)
    # recall for label 1
    recall_1 = np.nan if TP + FN == 0 else TP * 1.0 / (TP + FN)
    # precision for label 0
    precision_0 = np.nan if TN + FN == 0 else TN * 1.0 / (TN + FN)
    # recall for label 0
    recall_0 = np.nan if TN + FP == 0 else TN * 1.0 / (TN + FP)

    return precision_1, recall_1, precision_0, recall_0


def aupr(predicted_scores, true_labels):
    """
    Calculate AUPR[0], AUPR[1] and AUPR[hmean] for single experiment.
    :param predicted_scores: list of floats, list of predicted scores, i.e., 1 or 0.
    :param true_labels: list of ints, list of true labels, i.e., 1, 0.
    :return: list of floats, AUPR[0], AUPR[1], AUPR[hmean]
    """
    auprs = []

    # aupr for each class
    predicted_scores = np.array(predicted_scores)
    predicted_scores = np.vstack((1 - predicted_scores, predicted_scores))
    for i in range(2):
        p, r, th = precision_recall_curve(true_labels, predicted_scores[i,:], pos_label=i)
        aupr = auc(r, p)
        auprs.append(aupr)
    # hmean aupr
    if all(x > 0.0 for x in auprs):
        aupr_hmean = scipy.stats.hmean(auprs)
    else:
        aupr_hmean = 0.0

    return auprs[0], auprs[1], aupr_hmean



if __name__ == '__main__':
    true_labels = [1, 0, 1, 1, 0]
    predicted_labels = [1, 1, 1, 1, 0]
    predicted_scores = [0.9, 0.6, 0.7, 0.4, 0.2]

    # precision_1, recall_1, precision_0, recall_0 = calculate_PR(predicted_labels, true_labels)
    # print(precision_1, recall_1, precision_0, recall_0)

    metric_value_dict = aupr(predicted_scores, true_labels)
    print(metric_value_dict)

