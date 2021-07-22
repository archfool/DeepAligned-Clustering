#!/usr/bin bash

DATASET=clinc
MODEL_NAME=bert-base-uncased
#MODEL_NAME=roberta-large
for cl_sample_ratio in 0.1 0.2 0.3 0.5 0.7 0.9
do
  for batch_size in 8 16 32 64
  do
    for pretrain_epoch in 10
    do
      for seed in 0 1 2 3 4
      do
python DeepAligned.py \
    --data_dir /media/archfool/data/data/datasets-for-clustering \
    --save_results_path /media/archfool/data/data/my-sup-simcse-${MODEL_NAME}/result_log \
    --bert_model /media/archfool/data/data/huggingface/${MODEL_NAME} \
    --pretrain_dir /media/archfool/data/data/my-sup-simcse-${MODEL_NAME}/${DATASET} \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --known_cls_ratio 0.75 \
    --cluster_num_factor 1.0 \
    --labeled_ratio 0.1 \
    --wait_patient 20 \
    --num_pretrain_epochs ${pretrain_epoch} \
    --num_train_epochs 10 \
    --seed ${seed} \
    --freeze_bert_parameters \
    --save_model \
    --pretrain \
    --use_CL \
    --eval_per_epochs 1 \
    --cl_sample_ratio ${cl_sample_ratio} \
--model_name_or_path /media/archfool/data/data/huggingface/${MODEL_NAME} \
--pre_train_file /media/archfool/data/data/datasets-for-clustering/${DATASET}/pre_train_cl.tsv.csv \
--train_file /media/archfool/data/data/datasets-for-clustering/${DATASET}/train_cl.tsv.csv \
--output_dir /media/archfool/data/data/my-sup-simcse-${MODEL_NAME} \
--per_device_train_batch_size ${batch_size} \
--learning_rate 5e-5 \
--logging_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--eval_steps 999999999 \
--pooler_type avg \
--overwrite_output_dir \
--temp 0.05 \
--do_train \
--pad_to_max_length
      done
    done
  done
done
