# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/6/10 0010 下午 19:53
"""
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


def alignment(num_labels, centroids, km, feat_dim, device):
    if centroids is not None:

        old_centroids = centroids.cpu().numpy()
        new_centroids = km.cluster_centers_

        DistanceMatrix = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
        # linear_sum_assignment函数输入为cost矩阵。cost[i, j]表示工人i执行任务j所要花费的代价。
        # linear_sum_assignment函数输出row_ind和col_ind。row_ind表示选择哪几个工人，col_ind表示工人做哪个工作。
        row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

        new_centroids = torch.tensor(new_centroids).to(device)
        centroids = torch.empty(num_labels, feat_dim).to(device)

        alignment_labels = list(col_ind)
        for i in range(num_labels):
            label = alignment_labels[i]
            centroids[i] = new_centroids[label]

        pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
        pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

    else:
        centroids = torch.tensor(km.cluster_centers_).to(device)
        pseudo_labels = km.labels_

    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(device)

    return centroids, pseudo_labels


# def update_pseudo_labels(input_ids, input_mask, segment_ids, pseudo_labels, batch_size):
#     train_data = TensorDataset(input_ids, input_mask, segment_ids, pseudo_labels)
#     train_sampler = SequentialSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
#     return train_dataloader
