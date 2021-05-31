#!/usr/bin bash

    --model_name_or_path E:\data\huggingface\bert-base-uncased
    --train_file E:\data\datasets-for-simcse\nli_for_simcse.csv
    --output_dir E:\data\huggingface\simcse_model
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
    --do_train
    --do_eval

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/my-sup-simcse-bert-base-uncased \
    --num_train_epochs 3 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
