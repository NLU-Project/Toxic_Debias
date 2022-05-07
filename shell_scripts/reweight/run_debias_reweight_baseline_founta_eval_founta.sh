#!/bin/bash
export TOXIC_DIR=/scratch/sbp354/DSGA1012/Final_Project/data
export TASK_NAME=debias
export TRAIN_DATASET=founta/founta_train_shallow_remainder_random_0.01_seed42.csv
export TEACHER_DIR=/scratch/sbp354/DSGA1012/Final_Project/models/founta/roberta-large
export TEACHER_DATASET=finetune_founta/founta_train_shallow_remainder_random_0.01_seed42.csv_results.csv
export DEV_DATASET=founta
export LOSS=reweight_baseline

export MODEL_DIR=/scratch/sbp354/DSGA1012/Final_Project/models/founta_reweight_baseline

python /scratch/sbp354/DSGA1012/Final_Project/git/Toxic_Debias/run_toxic.py \
  --model_type roberta \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --save_steps 1000 \
  --logging_steps 1000 \
  --overwrite_output_dir \
  --overwrite_cache \
  --data_dir $TOXIC_DIR \
  --train_dataset $TRAIN_DATASET \
  --teacher_data_dir $TEACHER_DIR \
  --teacher_dataset $TEACHER_DATASET \
  --dev_dataset $DEV_DATASET \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --output_dir $MODEL_DIR \
  --mode $LOSS
