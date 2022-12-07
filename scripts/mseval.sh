#!/bin/bash
set -e

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PWD

METHODS=mlp.uspto-50k_rs
DATASET="rt-tpl-100"
EXP_ID=uspto-50k

if [ $# -eq 0 ]
  then
    echo "No argument supplied, using default targets rt-tpl-100"
  else
    METHODS=$1
    DATASET=$2
    EXP_ID=$3
fi

echo Starting model evaluation!
python retroeval/run_paroutes.py --methods $METHODS --data $DATASET --exp_id $EXP_ID

