#!/bin/bash
set -e

REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO

SSMODELS=mlp
DATASET=uspto-50k
METRICS="maxfrag_k,top_k,mrr,mss"
EXP_ID="uspto-50k"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using default models 'mlp', data uspto-50k"
  else
    SSMODELS=$1
    DATASET=$3
    EXP_ID=$2
fi

echo Starting model evaluation!
python retroeval/eval_single_step.py --models $SSMODELS --data $DATASET --exp_id $EXP_ID --metrics $METRICS