#!/usr/bin bash

DATASET=clinc
MODEL_NAME=bert-base-uncased
single_nova.py submit \
    --logdir /data/nfs/ruanzhihao322/log \
    --entry /data/nfs/ruanzhihao322/src/DeepAligned-Clustering/DeepAligned.py \
    --framework torch \
    --torch-version 1.5 \
    --gpu-per-work 1 \
    --pod-type 15,240 \
    --data_dir /data/nfs/ruanzhihao322/data/DeepAlignedClustering/ \
    --save_results_path /data/nfs/ruanzhihao322/model/DeepAlignedClustering/${DATASET} \
    --bert_model /data/nfs/ruanzhihao322/model/huggingface/${MODEL_NAME} \
    --pretrain_dir /data/nfs/ruanzhihao322/model/DeepAlignedClustering/${DATASET} \
    --dataset ${DATASET} \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --labeled_ratio 0.1 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --num_pretrain_epochs 50 \
    --num_train_epochs 50 \
    --seed 1234 \
    --freeze_bert_parameters \
    --pretrain
