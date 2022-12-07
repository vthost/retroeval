#!/bin/bash
set -e

conda create -y -n retroeval python=3.8 pytorch=1.13 cudatoolkit=11.1 rdkit tqdm -c pytorch -c conda-forge

eval "$(conda shell.bash hook)"
conda activate retroeval

pip install reaction-utils

# MULTI-STEP FUNCTIONALITY
python -m pip install git+https://github.com/MolecularAI/aizynthfinder.git@e8d22ab4afaa6fbeab260e375f91e195d4b17cb6
git clone https://github.com/MolecularAI/PARoutes.git

# OPTIONAL MODELS

# Chemformer
python -m pip install git+https://github.com/MolecularAI/Chemformer.git@b7cbe4cff871e00bf22888b0883cb1c147991530
python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git
mkdir -p models/chemformer
wget https://raw.githubusercontent.com/MolecularAI/Chemformer/main/bart_vocab.txt
mv bart_vocab.txt models/chemformer/
# Graphretro
pip install wandb # just that their code does not break
python -m pip install git+https://github.com/vsomnath/graphretro.git@4dfb03407e77c03d4dfcf8fe29761d8a12d4ae55
# Graph2SMILES - relative imports not working at runtime
#git clone https://github.com/coleygroup/Graph2SMILES.git
#conda install -y torchtext -c pytorch
#pip install gdown OpenNMT-py==1.2.0 networkx==2.5 selfies==1.0.3

# CHECKPOINTS from others

# Graphretro
#mkdir -p models/graphretro/SingleEdit_10-02-2021--08-44-37/checkpoints
#mkdir -p models/graphretro/LGIndEmbed_18-02-2021--12-23-26/checkpoints
#wget https://github.com/vsomnath/graphretro/blob/main/models/SingleEdit_10-02-2021--08-44-37/checkpoints/epoch_156.pt
#wget https://github.com/vsomnath/graphretro/blob/main/models/LGIndEmbed_18-02-2021--12-23-26/checkpoints/step_101951.pt
#mv epoch_156.pt SingleEdit_10-02-2021--08-44-37/checkpoints/
#mv step_101951.pt LGIndEmbed_18-02-2021--12-23-26/checkpoints/
# Graph2SMILES
#mkdir -p models/g2s
#wget -O USPTO_50k_dgcn.pt "https://drive.google.com/uc?id=11IJzTvyEvLclVSDN6gz2wy1Pg2hSOqa0&export=download&confirm=yes"
#mv USPTO_50k_dgcn.pt models/g2s/
