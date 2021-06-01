# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/5/31 0031 下午 21:24
"""

import os
import pandas as pd
import argparse
from dataloader import DatasetProcessor
import random

from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_examples(processor, args, data_dir, known_label_list):
    ori_examples = processor.get_examples(data_dir, "train")

    train_labels = np.array([example.label for example in ori_examples])
    train_labeled_ids = []

    # todo iter for every know_label, random choose a ratio of the corpus tobe labeled, and record their idx
    for label in known_label_list:
        num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
        pos = list(np.where(train_labels == label)[0])
        train_labeled_ids.extend(random.sample(pos, num))

    train_labeled_examples, train_unlabeled_examples = [], []
    for idx, example in enumerate(ori_examples):
        if idx in train_labeled_ids:
            train_labeled_examples.append(example)
        else:
            train_unlabeled_examples.append(example)

    return train_labeled_examples, train_unlabeled_examples


def generate_unsup_corpus():
    dir_input = r'E:\data\datasets-for-clustering\clinc'
    file_name_input = "train.tsv"
    dir_output = r'E:\data\datasets-for-clustering'
    file_name_output = "clinc_unsup_CL.txt"

    df = pd.read_csv(os.path.join(dir_input, file_name_input), sep='\t')
    corpus = df['text'].to_list()
    with open(os.path.join(dir_output, file_name_output), 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus))


def generate_sup_corpus():
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
    args.data_dir = r'/media/archfool/data/data/datasets-for-clustering'

    set_seed(args.seed)
    max_seq_lengths = {'clinc': 30, 'stackoverflow': 45, 'banking': 55}
    args.max_seq_length = max_seq_lengths[args.dataset]

    processor = DatasetProcessor()
    data_dir = os.path.join(args.data_dir, args.dataset)
    # todo judge 3 kinds of labels
    all_label_list = processor.get_labels(data_dir)
    n_known_cls = round(len(all_label_list) * args.known_cls_ratio)
    known_label_list = list(np.random.choice(np.array(all_label_list), n_known_cls, replace=False))
    num_labels = int(len(all_label_list) * args.cluster_num_factor)

    train_labeled_examples, train_unlabeled_examples = get_examples(processor, args, data_dir, known_label_list)

    corpus = []
    for label in known_label_list:
        example_set = []
        for example in train_labeled_examples:
            if example.label==label:
                example_set.append(example.text_a)

        if len(example_set)<=1:
            continue

        for text_a in example_set:
            for text_b in example_set:
                if text_a!=text_b:
                    corpus.append(text_a+'\t'+text_b)

    random.shuffle(corpus)
    corpus = ["sent0\tsent1"] + corpus

    dir_output = "/media/archfool/data/data/datasets-for-clustering"
    file_name_output = "clinc_sup_CL.tsv.csv"
    with open(os.path.join(dir_output, file_name_output), 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus))


if __name__ == "__main__":
    generate_sup_corpus()
    generate_unsup_corpus()

    print("END")
