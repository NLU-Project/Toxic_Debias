#!/bin/bash
export TOXIC_DIR=../Final_Project/data/founta
export TRAIN_DATASET=founta
export MODE=random
export PERCENT=.005

python ../Toxic_Debias/src/shallow_subsample.py \
  --overwrite_output_dir \
  --data_dir $TOXIC_DIR \
  --train_dataset $TRAIN_DATASET \
  --output_dir $TOXIC_DIR \
  --mode $MODE \
  --sample_percent $PERCENT