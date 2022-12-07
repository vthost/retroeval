#!/bin/bash
set -e

REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO:$REPO/Graph2SMILES

POLICY=mlp.uspto-50k
DATASET="rt-tpl-100"
STOCK=rt-tpl-100
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using default arguments: mlp, uspto-50k, rt-tpl-100"
  else
    POLICY=$1
    DATASET=$2
    STOCK=$3
fi

echo Starting model run!
python retroeval/run_aizynthfinder.py --policy $POLICY --config "./assets/aizynthfinder/config.yml" \
                              --data $DATASET --stock $STOCK
