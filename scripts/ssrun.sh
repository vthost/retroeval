#!/bin/bash
set -e

REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO:$REPO/Graph2SMILES

SSMODEL=mlp
CHECKPOINT=uspto-50k
DATASET=uspto-50k
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using default model mlp, data uspto-50k."
  else
    SSMODEL=$1
    CHECKPOINT=$2
    DATASET=$3
fi

echo Starting model run!
python retroeval/run_single_step.py --model $SSMODEL --checkpoint $CHECKPOINT --data $DATASET
