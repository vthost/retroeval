from rdkit import Chem
import pandas as pd

from retroeval.utils.mol import canonicalize


# run directly on cluster to get pickle protocol from molbart env
def create_chemformer_data_uspto50k():
    ps, rs, ss = [], [], []
    for part in ['train', 'eval', 'test']:
        p = f"./graphretro/datasets/uspto-50k/canonicalized_{part}.csv"
        data = pd.read_csv(p)["reactants>reagents>production"]
        for d in data:
            d = d.strip().split(">")
            ps += [Chem.MolFromSmiles(canonicalize(d[-1]))]
            rs += [Chem.MolFromSmiles(canonicalize(d[0]))]
            ss += ["valid"] if part == "eval" else [part]

    df = {'reactants_mol': rs,
          'products_mol': ps,
          'reaction_type': ['<UNK>'] * len(ps),
          'set': ss}
    df = pd.DataFrame(data=df)
    df.to_pickle(f"./assets/chemformer/uspto50k.pickle")


def create_chemformer_data_r(fid):
    ps, rs, ss = [], [], []
    for part in ['train', 'valid', 'test']:
        p = f"./data/{fid}/{fid}-{part}-rxns.txt"
        data = pd.read_csv(p, usecols=["product", "reactants"])
        for i, r in data.iterrows():
            ps += [Chem.MolFromSmiles(canonicalize(r["product"]))]
            rs += [Chem.MolFromSmiles(canonicalize(r["reactants"]))]
            ss += [part]

    df = {'reactants_mol': rs,
          'products_mol': ps,
          'reaction_type': ['<UNK>'] * len(ps),
          'set': ss}
    df = pd.DataFrame(data=df)
    df.to_pickle(f"./assets/chemformer/{fid}.pickle")


if __name__ == "__main__":
    for fid in ["rt", "rd"]:
        create_chemformer_data_r(fid)
