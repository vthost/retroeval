#!/bin/bash
set -e

#CHECKPOINTS FROM US
CHECKPOINTS=(  \
"mlp.tar.gz"  \
"mlppp.tar.gz"  \
"chemformer.tar.gz" \
)

mkdir -p models
cd models

for MODEL in ${MODELS[*]}; do
  wget "https://neurips.s3.us-east.cloud-object-storage.appdomain.cloud/ai4science_retroval/models/${MODEL}"
  tar -xf $MODEL
  rm $MODEL
done

# CHECKPOINTS FROM OTHERS
# Graphretro
mkdir -p graphretro/SingleEdit_10-02-2021--08-44-37/checkpoints
mkdir -p graphretro/LGIndEmbed_18-02-2021--12-23-26/checkpoints
wget https://github.com/vsomnath/graphretro/blob/main/models/SingleEdit_10-02-2021--08-44-37/checkpoints/epoch_156.pt
wget https://github.com/vsomnath/graphretro/blob/main/models/LGIndEmbed_18-02-2021--12-23-26/checkpoints/step_101951.pt
mv epoch_156.pt graphretro/SingleEdit_10-02-2021--08-44-37/checkpoints/
mv step_101951.pt graphretro/LGIndEmbed_18-02-2021--12-23-26/checkpoints/

# Graph2SMILES
#mkdir -p g2s
#wget -O USPTO_50k_dgcn.pt "https://drive.google.com/uc?id=11IJzTvyEvLclVSDN6gz2wy1Pg2hSOqa0&export=download&confirm=yes"
#mv USPTO_50k_dgcn.pt g2s/



