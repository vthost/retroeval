#!/bin/bash
set -e

REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO

SSMODELS=mlp
DATASET=uspto-ms
METRICS="r_k"
EXP_ID="uspto-50k"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using default models 'mlp', data uspto-ms"
  else
    SSMODELS=$1
    DATASET=$2
    EXP_ID=$3
fi

echo Starting model evaluation!
python retroeval/eval_multi_sol.py --models $SSMODELS --data $DATASET --exp_id $EXP_ID --metrics $METRICS