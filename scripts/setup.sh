#!/bin/bash
set -e

conda create -y -n retroeval python=3.8 pytorch=1.13 pytorch-cuda=11.7 -c pytorch -c nvidia

eval "$(conda shell.bash hook)"
conda activate retroeval

# force this channel to get it for python 3.8
conda install -y -c conda-forge rdkit

pip install reaction-utils

# MULTI-STEP FUNCTIONALITY

python -m pip install git+https://github.com/MolecularAI/aizynthfinder.git@e8d22ab4afaa6fbeab260e375f91e195d4b17cb6
git clone https://github.com/MolecularAI/PARoutes.git

# OPTIONAL MODELS

# Chemformer
#python -m pip install git+https://github.com/MolecularAI/Chemformer.git@b7cbe4cff871e00bf22888b0883cb1c147991530
#python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git
#mkdir -p models/chemformer
#wget https://raw.githubusercontent.com/MolecularAI/Chemformer/main/bart_vocab.txt
#mv bart_vocab.txt models/chemformer/

# Graphretro
#pip install wandb # just that their code does not break
#python -m pip install git+https://github.com/vsomnath/graphretro.git@4dfb03407e77c03d4dfcf8fe29761d8a12d4ae55

# Graph2SMILES - relative imports not working at runtime
#git clone https://github.com/coleygroup/Graph2SMILES.git
#conda install -y torchtext -c pytorch
#pip install gdown OpenNMT-py==1.2.0 networkx==2.5 selfies==1.0.3

