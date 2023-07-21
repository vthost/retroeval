import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict, Counter, defaultdict
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit import Chem
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

from retroeval.utils.mol import canonicalize
from retroeval.utils.data import load_dataset
from reaction_utils import *


# NOTE
# you need "1976_Sep2016_USPTOgrants_smiles.rsmi" in this directory
graphretropath = None  # set this


def read_uspto50k(name):
    p = f"{graphretropath}/graphretro/datasets/uspto-50k/canonicalized_{name}.csv"
    data = pd.read_csv(p)["reactants>reagents>production"]
    ps, rs = [], []
    for d in data:
        d = d.strip().split(">")
        ps += [canonicalize(d[-1])]
        rs += [canonicalize(d[0])]
    return ps, rs


if __name__ == "__main__":

    name = "1976_Sep2016_USPTOgrants_smiles.rsmi"
    df = pd.read_csv(name, sep='\t', usecols=["ReactionSmiles", "PatentNumber"])
    l0 = len(df)
    print(l0)
    # 1808937

    # canonicalize
    df.ReactionSmiles = df.ReactionSmiles.apply(sp)
    df = df.drop(df[df.ReactionSmiles.values == None].index)
    print(f'> {l0-len(df)} reactions can\'t be canonicalized')
    # > 643 reactions can't be canonicalized

    # create ids
    df["PatentNumber"] = df.PatentNumber + ';' + df.index.map(str)
    df = df.to_records(index=False)

    with open('data_canon.pkl', 'wb') as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    ct = 0
    # group reactions by target
    mor, dup = 0, 0
    data = defaultdict(lambda: defaultdict(list))
    for e, id_str in df.tolist():
        if e is None:
            ct += 1
            continue
        (product, reactants) = e
        if '.' not in product:
            data[product][reactants] += [id_str]
            if len(data[product][reactants]) > 1:
                dup += 1
        else:
            mor += 1

    print(len(data))
    print(f'> {ct} reactions None')
    print(f'> {mor} reactions have more than one product')
    print(f'> {dup} reactions have more than one origin')
    print_distribution(data)

    # can't pickle lambda from default dict, so not saving here intermediately

    # 984707
    # > 0 reactions None
    # > 142332 reactions have more than one product
    # > 676824 reactions have more than one origin
    # {1: 980649, 2: 3799, 3: 198, 4: 40, 5: 12, 6: 4, 8: 2, 9: 1, 13: 1, 15: 1}

    l0 = len(data)
    data = dict(filter(lambda pair: len(pair[1]) > 1, data.items()))
    print(f'> {l0-len(data)} products have only one reactant set')
    print(f'> {len(data)} remaining.')
    # > 980649 products have only one reactant set
    # > 4058 remaining.

    l0 = len(data)
    data = am(data)
    print_distribution(data)
    # {2: 3506, 3: 148, 4: 99, 5: 35, 6: 17, 7: 5, 8: 5, 9: 3, 10: 3, 12: 1, 13: 2, 14: 2, 16: 1, 18: 1, 22: 2, 27: 1,
    # 28: 1, 29: 1}

    l0 = len(data)  # might be possible that am merged also reactants
    data = dict(filter(lambda pair: len(pair[1]) > 1, data.items()))
    print(f'> {l0-len(data)} products have only one reactant set')
    print(f'> {len(data)} remaining.')

    with open('data_amrem.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)  #can't pickle lambda from default dict

    # additional filtering
    uspto_products = read_uspto50k('train')[0] + read_uspto50k('eval')[0]
    torch.save(uspto_products, "uspto_products.pt")
    # uspto_products = torch.load("uspto_products.pt")
    l0 = len(data)
    data = dict(filter(lambda pair: pair[0] not in uspto_products
                                    and 6 >= len(pair[1]) > 1, data.items()))
    print(f'> {l0-len(data)} reactions occur in 50k or have more than 6 solutions')

    print_distribution(data)
    print(len(data))
    # > 0 products have only one reactant set
    # > 3833 remaining.
    # > 332 reactions occur in 50k or have more than 6 solutions
    # {2: 3247, 3: 130, 4: 82, 5: 26, 6: 16}
    # 3501

    with open('uspto-ms-pids.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    data = {k: [vi[0] for vi in v] for k, v in data.items()}
    with open('uspto-ms.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
