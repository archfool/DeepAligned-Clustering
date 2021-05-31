#!/usr/bin bash

    --model_name_or_path /media/archfool/data/data/huggingface/bert-base-uncased
    --train_file /media/archfool/data/data/datasets-for-clustering/clinc_unsup_CL.txt
    --output_dir /media/archfool/data/data/huggingface/my-unsup-simcse-bert-base-uncased
    --num_train_epochs 20
    --per_device_train_batch_size 64
    --learning_rate 3e-5
    --max_seq_length 32
    --evaluation_strategy steps
    --metric_for_best_model stsb_spearman
    --load_best_model_at_end
    --eval_steps 100
    --pooler_type cls
    --mlp_only_train
    --overwrite_output_dir
    --temp 0.05
    --do_train
    --do_eval

    --model_name_or_path E:\data\huggingface\bert-base-uncased
    --train_file E:\data\datasets-for-clustering\clinc_unsup_CL.txt
    --output_dir E:\data\huggingface\simcse_model\my-unsup-simcse-bert-base-uncased
    --num_train_epochs 200
    --per_device_train_batch_size 64
    --learning_rate 3e-5
    --max_seq_length 32
    --evaluation_strategy steps
    --metric_for_best_model stsb_spearman
    --load_best_model_at_end
    --eval_steps 125
    --pooler_type cls
    --mlp_only_train
    --overwrite_output_dir
    --temp 0.05
    --do_train
    --do_eval


python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
