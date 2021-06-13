# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/6/10 0010 下午 19:53
"""
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score


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

        # update cluster centrol vector
        alignment_labels = list(col_ind)
        for i in range(num_labels):
            label = alignment_labels[i]
            centroids[i] = new_centroids[label]

        # update label of samples, coresponding to the first one
        pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
        pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

    else:
        centroids = torch.tensor(km.cluster_centers_).to(device)
        pseudo_labels = km.labels_

    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(device)

    return centroids, pseudo_labels


def evaluation(trainer, data):

    # feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
    # feats = feats.cpu().numpy()
    # km = KMeans(n_clusters=self.num_labels).fit(feats)
    feats, labels = trainer.get_featureEmbd_label(data.test_dataloader)
    feats = feats.cpu().numpy()
    km = KMeans(n_clusters=data.num_labels).fit(feats)

    y_pred = km.labels_
    y_true = labels.cpu().numpy()

    results = clustering_score(y_true, y_pred)
    print('results', results)

    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred = np.array([map_[idx] for idx in y_pred])

    cm = confusion_matrix(y_true, y_pred)
    print('confusion matrix\n', cm)

    # save_results(results)
    return results


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2)}


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc




# def update_pseudo_labels(input_ids, input_mask, segment_ids, pseudo_labels, batch_size):
#     train_data = TensorDataset(input_ids, input_mask, segment_ids, pseudo_labels)
#     train_sampler = SequentialSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
#     return train_dataloader
