# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/5/31 0031 下午 21:24
"""

import os
import pandas as pd
import argparse

from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
import time

def generate_unsup_corpus():
    dir_input = r'E:\data\datasets-for-clustering\clinc'
    file_name_input = "train.tsv"
    dir_output = r'E:\data\datasets-for-clustering'
    file_name_output = "clinc_unsup_CL.txt"

    df = pd.read_csv(os.path.join(dir_input, file_name_input), sep='\t')
    corpus = df['text'].to_list()
    with open(os.path.join(dir_output, file_name_output), 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")
    parser.add_argument("--dataset", default=None, type=str,
                        help="The name of the dataset to train selected.")
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--known_cls_ratio", default=0.75, type=float,
                        help="The number of known classes.")
    parser.add_argument("--cluster_num_factor", default=1.0, type=float,
                        help="The factor (magnification) of the number of clusters K.")
    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples in the training set.")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    args = parser.parse_args()
    args.seed = 1234
    args.dataset = 'clinc'
    args.data_dir = r'E:\data\datasets-for-clustering'
    data = Data(args)

    generate_unsup_corpus()

    print("END")
