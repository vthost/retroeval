import re
import pickle as pickle
import pandas as pd

from retroeval.utils.mol import canonicalize
from retroeval.utils.config import *


def load_multisol_dataset(name, return_dict=False, canonical=True):
    assert name in MULTI_SOL_DATASET_INFO
    data_info = MULTI_SOL_DATASET_INFO[name]
    path = f'{data_info[DI_COL_PATH]}/{data_info[DI_COL_FILE]}'
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if canonical:
        data = {canonicalize(prod): [canonicalize(rs) for rs in reacts] for prod, reacts in data.items()}

    if return_dict:
        return data
    return list(data.keys()), [list(v) for v in data.values()]


def _load_registered_dataset(data_info):
    path = f'{data_info[DI_COL_PATH]}/{data_info[DI_COL_FILE]}'
    df = pd.read_csv(path)

    if data_info[DI_COL_RXN] is None:
        prod_col, reactants_col = data_info[DI_COL_PROD], data_info[DI_COL_REACT]
        return df[prod_col].values.tolist(), df[reactants_col].values.tolist()

    df = df[data_info[DI_COL_RXN]].apply(lambda rxn: rxn.split('>'))
    return [rxn[-1] for rxn in df], [rxn[0] for rxn in df]


# our default format
def _load_dataset(name, part):
    path = f'./data/{name}/{name}-{part}-rxns.csv'
    prod_col, reactants_col = 'product', 'reactants'
    df = pd.read_csv(path)
    return df[prod_col].values.tolist(), df[reactants_col].values.tolist()


def load_dataset(name, canonical=True, part='test'):
    if name in DATASET_INFO:
        prods, reacts = _load_registered_dataset(DATASET_INFO[name][part])
    else:
        # our format, but may not be give explicitly ie we also allow for just the name and then assume 'test'
        for _part in ["train", "valid", "test"]:
            if name.endswith(_part):
                part = _part
                name = name[:name.rindex('-')]
                break
        prods, reacts = _load_dataset(name, part)

    if canonical:
        prods, reacts = [canonicalize(p) for p in prods], [canonicalize(rs) for rs in reacts]

    return prods, reacts


def load_templates_from_dfs(df_dict, tpl_col, tpl_parts=["train"], index=True):

    s = pd.DataFrame(columns=[tpl_col])[tpl_col]
    for part in tpl_parts:
        s = s.append(df_dict[part][tpl_col])
    s = list(s)

    # need to cope with potential difference in mappings
    # https://github.com/coleygroup/rxn-ebm/blob/1919eeccdd31e16ec7a44478b756bcd974c35a3c/rxnebm/data/preprocess/clean_smiles.py#L241
    # lookahead (?=]) to ensure it's the decimal from an atom mapping
    s_u = [re.sub(':\d+(?=])', '', r) for r in s]
    tpl_idx0, s_u_fact = pd.factorize(s_u)
    tpls = [s[s_u.index(u)] for u in s_u_fact]

    if not index:
        return tpls

    tpl_idx, i = {}, 0
    for part in tpl_parts:
        tpl_idx[part] = list(tpl_idx0[i:i+len(df_dict[part])])
        i += len(df_dict[part])

    tpls = {tpl: i for i, tpl in enumerate(tpls)}
    unk = len(tpls)
    for part in ["train", "valid", "test"]:  # train makes not so much sense
        if part in tpl_parts or part not in df_dict:
            continue
        tpl_idx[part] = [tpls.get(tpl, unk) for tpl in df_dict[part]['retro_template']]

    return list(tpls.keys()), tpl_idx


def load_templates(dataset, parts=["train", "valid"]):

    if dataset in DATASET_INFO:
        data_info = DATASET_INFO[dataset]
        dfs = {part: pd.read_csv(f'{data_info[part][DI_COL_PATH]}/{data_info[part][DI_COL_FILE]}') for part in parts}
        tpl_col = data_info[parts[0]][DI_COL_TPL]  # we assume all parts have the same column
    else:
        dfs = {part: pd.read_csv(f'./data/{dataset}/{dataset}-{part}-rxns.csv') for part in parts}
        tpl_col = 'retro_template'

    tpls = load_templates_from_dfs(dfs, tpl_col, tpl_parts=parts, index=False)
    return tpls


