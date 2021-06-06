#!/usr/bin bash

    --data_dir /media/archfool/data/data/datasets-for-clustering
    --save_results_path /media/archfool/data/data/my-sup-simcse-bert-base-uncased/${DATASET}
    --bert_model /media/archfool/data/data/huggingface/${MODEL_NAME}
    --pretrain_dir /media/archfool/data/data/my-sup-simcse-bert-base-uncased/${DATASET}
    --dataset ${DATASET}
    --known_cls_ratio 0.75
    --cluster_num_factor 1.0
    --labeled_ratio 0.1
    --num_pretrain_epochs 50
    --num_train_epochs 50
    --seed 1234
    --freeze_bert_parameters
    --pretrain
    --save_model


    --data_dir /data/nfs/ruanzhihao322/data/DeepAlignedClustering/ \
    --save_results_path /data/nfs/ruanzhihao322/model/DeepAlignedClustering/${DATASET} \
    --pretrain_dir /data/nfs/ruanzhihao322/model/DeepAlignedClustering/${DATASET} \
    --bert_model /data/nfs/ruanzhihao322/model/${MODEL_NAME} \
    --freeze_bert_parameters \
    --save_model \
    --pretrain \
    --dataset ${DATASET} \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1 \
    --seed 1234 \
    --labeled_ratio 0.1 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --wait_patient 20 \
    --num_pretrain_epochs 50 \
    --num_train_epochs 50 \




for s in 0 1 2 3 4 5 6 7 8 9
do 
    python DeepAligned.py \
        --dataset clinc \
        --known_cls_ratio 0.75 \
        --cluster_num_factor 1 \
        --seed $s \
        --freeze_bert_parameters \
        --pretrain
done
