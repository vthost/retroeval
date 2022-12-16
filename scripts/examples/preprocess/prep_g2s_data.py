import os
import pandas as pd
from retroeval.utils.mol import canonicalize


def create_g2s_data_r(fid):
    ps, rs, ss = [], [], []
    for part in ['train', 'valid', 'test']:
        p = f"./data/{fid}/{fid}-{part}-rxns.txt"
        data = pd.read_csv(p, usecols=["product", "reactants"])
        for i, r in data.iterrows():
            ps += [canonicalize(r["product"])]
            rs += [canonicalize(r["reactants"])]

        with open(f"./Graph2SMILES/data/{fid}/src-{part}.txt") as f:
            f.write("\n".join(ps))
        with open(f"./Graph2SMILES/data/{fid}/tgt-{part}.txt") as f:
            f.write("\n".join(rs))


if __name__ == "__main__":
    for fid in ["rt", "rd"]:
        os.makedirs(f"./Graph2SMILES/data/{fid}", exist_ok=True)
        create_g2s_data_r(fid)
