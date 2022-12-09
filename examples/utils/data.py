import pandas as pd
import torch

from retroeval.utils.data import load_templates_from_dfs


def load_dataset(name, tpls=False, valid_tpls=True):
    df = {}
    prods, reacts = {}, {}
    for part in ['train', 'valid', 'test']:
        path = f'./data/{name}/{name}-{part}-rxns.csv'
        df[part] = pd.read_csv(path, usecols=['product', 'reactants', 'retro_template'])

        prods[part] = df[part]['product'].values.tolist()
        reacts[part] = df[part]['reactants'].values.tolist()

    if not tpls:
        return prods, reacts

    tpls, tpl_idx = load_templates_from_dfs(df, 'retro_template', index=True,
                                            tpl_parts=["train"] + (["valid"] if valid_tpls else []))

    return prods, reacts, tpl_idx, tpls
