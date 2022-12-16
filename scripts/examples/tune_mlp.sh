#!/bin/bash
set -e

REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO

DATA="uspto-50k"
MODEL="examples.models.mlp.RetroMLP"
PATIENCE=50
BATCH=1024
LRS=(5e-4)  #1e-3 5e-4 1e-4 5e-3)
CONFIG_DIR="examples/configs_all"

python scripts/examples/preprocess/generate_configs.py --directory $CONFIG_DIR
echo generated configs

for LR in "${LRS[@]}"
do
  for MODEL_CONFIG in $CONFIG_DIR/*
  do
    echo ">>" lr $LR, config $MODEL_CONFIG

    python examples/train/train.py --model  $MODEL --model_config $MODEL_CONFIG --data $DATA \
            --save_dir "" --batch_size $BATCH --lr $LR --wd 0.00 --patience $PATIENCE --train_with_valid=0 --cache=0 #--verbose 0

    echo ">>" lr $LR, config $MODEL_CONFIG completed
  done
done



