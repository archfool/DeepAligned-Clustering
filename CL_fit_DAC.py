# -*- coding: utf-8 -*-
"""
@author: ruanzhihao
Created on 2021/6/2 0002 下午 14:06
"""
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple


@dataclass
class DacArguments:
    data_dir: str = field(
        default="data",
        metadata={"help": "The input data dir. Should contain the .csv files (or other data files) for the task."},
    )

    save_results_path: str = field(
        default="outputs",
        metadata={"help": "The path to save results."},
    )

    pretrain_dir: str = field(
        default="pretrain_models",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    bert_model: str = field(
        default="uncased_L-12_H-768_A-12",
        metadata={"help": "The path for the pre-trained bert model."},
    )

    # max_seq_length: int = field(
    #     default=None,
    #     metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
    #                       "than this will be truncated, sequences shorter will be padded."},
    # )

    feat_dim: int = field(
        default=768,
        metadata={"help": "The feature dimension."},
    )

    warmup_proportion: float = field(
        default=0.1,
        metadata={},
    )

    freeze_bert_parameters: bool = field(
        default=None,
        metadata={
            # "action": "store_true",
            "help": "Freeze the last parameters of BERT."},
    )

    save_model: bool = field(
        default=None,
        metadata={
            # "action": "store_true",
            "help": "Save trained model."},
    )

    pretrain: bool = field(
        default=None,
        metadata={
            # "action": "store_true",
            "help": "Pre-train the model with labeled data."},
    )

    dataset: str = field(
        default=None,
        metadata={
            "required":  True,
            "help": "The name of the dataset to train selected."},
    )

    known_cls_ratio: float = field(
        default=0.75,
        metadata={
            "required":  True,
            "help": "The number of known classes."},
    )

    cluster_num_factor: float = field(
        default=1.0,
        metadata={
            "required":  True,
            "help": "The factor (magnification) of the number of clusters K."},
    )

    seed_dac: int = field(
        default=1234,
        metadata={"help": "Random seed for initialization."},
    )

    # seed: int = field(
    #     default=0,
    #     metadata={"help": "Random seed for initialization."},
    # )

    method: str = field(
        default='DeepAligned',
        metadata={"help": "Which method to use."},
    )

    labeled_ratio: float = field(
        default=0.1,
        metadata={"help": "The ratio of labeled samples in the training set."},
    )

    gpu_id: str = field(
        default="0",
        metadata={"help": "Select the GPU id."},
    )

    train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for training."},
    )

    eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for evaluation."},
    )

    wait_patient: int = field(
        default=20,
        metadata={"help": "Patient steps for Early Stop."},
    )

    num_pretrain_epochs: float = field(
        default=100,
        metadata={"help": "The pre-training epochs."},
    )

    num_train_epochs_dac: float = field(
        default=100,
        metadata={"help": "The training epochs."},
    )

    # num_train_epochs: float = field(
    #     default=100,
    #     metadata={"help": "The training epochs."},
    # )

    lr_pre: float = field(
        default=5e-5,
        metadata={"help": "The learning rate for pre-training."},
    )

    lr: float = field(
        default=5e-5,
        metadata={"help": "The learning rate for training."},
    )

    use_CL: bool = field(
        default=None,
        metadata={
            # "action": "store_true",
            "help": "use Contrastive Learning."},
    )

