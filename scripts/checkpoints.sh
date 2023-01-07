#!/bin/bash
set -e

DATASETS=(  \
"rt-1k.tar.gz"  \
"rd-1k.tar.gz"  \
"rt-tpl-100.tar.gz"  \
#"rt.tar.gz"  \
#"rd.tar.gz"  \
)

mkdir -p data
cd data

for DATASET in ${DATASETS[*]}; do
  wget "https://neurips.s3.us-east.cloud-object-storage.appdomain.cloud/ai4science_retroval/data/${DATASET}"
  tar -xf $DATASET
  rm $DATASET
done

