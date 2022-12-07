#!/bin/bash
set -e

REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO

MODEL="examples.models.mlp.RetroMLP"
MODEL_CONFIG="./examples/configs/mlp_default.json"
DATA="uspto-50k"
SAVE="./models/mlp/"
BATCH=1024
LR=0.0005
PATIENCE=100


python examples/train/train.py --model  $MODEL --model_config $MODEL_CONFIG --data $DATA \
            --save_dir $SAVE --batch_size $BATCH --lr $LR --patience $PATIENCE