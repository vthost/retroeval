#!/bin/bash
# according to: https://github.com/MolecularAI/PaRoutes/tree/main/setup
# You need to download the raw data as described in that repository first
set -e

cd setup
REPO=$PWD

source activate retroeval
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$REPO:{YOUR-PATH}/retroeval/


echo run "extract_uspto_data"
python extract_uspto_data.py --filename ../data/paroutes/uspto_raw_template_library.csv

echo run "extract_routes"
python extract_routes.py --max-workers 32

echo run "analyze"
python analyze_routes.py

echo run "overlaps"
python find_non_overlaps.py --output uspto/non_overlapping_routes.pickle

# uncomment one of the below
#echo run "select"
#python select_routes_time.py

#python select_routes_div.py
