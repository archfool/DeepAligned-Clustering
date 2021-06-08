#!/usr/bin bash

DATASET=clinc
MODEL_NAME=bert-base-uncased

    --data_dir /media/archfool/data/data/datasets-for-clustering
    --save_results_path /media/archfool/data/data/my-sup-simcse-bert-base-uncased/${DATASET}
    --bert_model /media/archfool/data/data/huggingface/${MODEL_NAME}
    --pretrain_dir /media/archfool/data/data/my-sup-simcse-bert-base-uncased/${DATASET}
    --dataset ${DATASET}
    --known_cls_ratio 0.75
    --cluster_num_factor 1.0
    --labeled_ratio 0.1
    --wait_patient 20
    --num_pretrain_epochs 1
    --num_train_epochs 50
    --seed 1234
    --freeze_bert_parameters
    --pretrain
    --save_model
    --use_CL
    --eval_epochs 1
--model_name_or_path /media/archfool/data/data/huggingface/bert-base-uncased
--train_file /media/archfool/data/data/datasets-for-clustering/${DATASET}/train_cl.tsv.csv
--output_dir /media/archfool/data/data/my-sup-simcse-bert-base-uncased
--per_device_train_batch_size 128
--learning_rate 5e-5
--max_seq_length 32
--evaluation_strategy steps
--eval_steps 999999999
--pooler_type cls
--overwrite_output_dir
--temp 0.05
--seed 1234
--do_train

--load_best_model_at_end
--metric_for_best_model stsb_spearman
--do_eval


    --data_dir /data/nfs/ruanzhihao322/data/DeepAlignedClustering/
    --save_results_path E:\data\my-sup-simcse-bert-base-uncased\${DATASET}
    --bert_model /data/nfs/ruanzhihao322/model/${MODEL_NAME}
    --pretrain_dir E:\data\my-sup-simcse-bert-base-uncased\${DATASET}
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
    --use_CL
--model_name_or_path E:\data\my-unsup-simcse-bert-base-uncased
--train_file E:\data\datasets-for-clustering\clinc_sup_CL.tsv.csv
--output_dir E:\data\my-sup-simcse-bert-base-uncased
--num_train_epochs 1
--per_device_train_batch_size 8
--learning_rate 5e-5
--max_seq_length 32
--evaluation_strategy steps
--metric_for_best_model stsb_spearman
--load_best_model_at_end
--eval_steps 125
--pooler_type cls
--overwrite_output_dir
--temp 0.05
--seed 1234
--do_train
--do_eval

