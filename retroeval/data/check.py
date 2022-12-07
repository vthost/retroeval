
import argparse
import numpy as np
import os
import torch
import pickle

import pandas as pd


def simple_dataset_check(name, rxn_col):
    df = {}
    for part in ['train', 'valid', 'test']:
        path = f'./data/{name}/{name}-{part}-rxns.csv'
        df[part] = pd.read_csv(path)

        l1 = len(df)
        df2 = df[part].drop_duplicates(subset=rxn_col, keep="first")
        l2 = len(df2)
        assert l1 == l2

    ct = 0
    for rxn in df['test']:
        if rxn in df['train'] + df['valid']:
            ct += 1
    print("test rxns CRITICAL", ct)
    ct = 0
    for rxn in df['valid']:
        if rxn in df['train']:
            ct += 1
    print("valid rxns CRITICAL", ct)


if __name__ == "__main__":
    for dataset in os.listdir("./data"):
        if "-ms" in dataset:
            continue
        simple_dataset_check(dataset, "smiles")
