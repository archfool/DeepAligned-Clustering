import itertools
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import copy
import torch.nn.functional as F
import random
import csv
import sys
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import time


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2)}


def save_result(result, args):
    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    # local_time = time.strftime("%m-%d %H:%M:%S", time.localtime())
    # var = [local_time, args.dataset, args.method, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor,
    #        args.seed]
    # names = ['datetime', 'dataset', 'method', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed']
    # vars_dict = {k: v for k, v in zip(names, var)}
    vars_dict = {
        'datetime': time.strftime("%m-%d %H:%M:%S", time.localtime()),
        'dataset': args.dataset,
        'method': args.method,
        'seed': args.seed,
        'known_cls_ratio': args.known_cls_ratio,
        'cluster_num_factor': args.cluster_num_factor,
        'labeled_ratio': args.labeled_ratio,
        'pretrain_epoch': args.num_pretrain_epochs,
        'train_epoch': args.num_train_epochs,
        'cl_sample_ratio': args.cl_sample_ratio,
        'batch_size': args.per_device_train_batch_size,
        'lr': args.learning_rate,
        'pooler_type': args.pooler_type,
    }
    results = dict(result, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = str(args.model_name) + '_results_log' + '.csv'
    results_path = os.path.join(args.save_results_path, file_name)

    if not os.path.exists(results_path):
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori, columns=keys)
        df1.to_csv(results_path, index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results, index=[1])
        df1 = df1.append(new, ignore_index=True)
        df1.to_csv(results_path, index=False)
    data_diagram = pd.read_csv(results_path)

    # print('test_results', data_diagram)
