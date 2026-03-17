#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric
from medpy.metric import hd95


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


# Metrics to track training performance
def compute_dice(output, label):
    intersection = (output * label).sum(dim=(1, 2, 3))
    dice_score = (2. * intersection) / (output.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)))
    return dice_score

def compute_jaccard(output, label):
    intersection = (output * label).sum(dim=(1, 2, 3))
    union = output.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) - intersection
    jaccard_score = intersection / union
    return jaccard_score

def compute_hd95(pred, target, max_dist):
    hd95_scores = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for p, t in zip(pred, target):
        if np.sum(p) == 0 or np.sum(t) == 0:
            hd95_scores.append(max_dist)  # Return max distance if either set is empty
        else:
            try:
                hd95_scores.append(hd95(p, t))
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                hd95_scores.append(max_dist)
    return hd95_scores

