# utilities for cleaning, based upon GLN's clean_uspto.py
import re
import pickle
from tqdm import tqdm
import numpy as np
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from rdkit import Chem
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')
from collections import OrderedDict, Counter


def print_distribution(r_lst):
    occurance = {}
    for p in r_lst:
        occurance[p] = len(r_lst[p])
    v = list(occurance.values())

    hist = dict(Counter(v))
    hist = dict(OrderedDict(sorted(hist.items())))
    print(hist)


# keep atom mapping
def simple_canonicalize(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(m)
    except Exception as e:
        return None
    return smiles


def get_contrib_reactants_smiles(prod, reactants):
    # Get rid of reactants when they don't contribute to this prod
    reactants = [Chem.MolFromSmiles(i) for i in reactants]

    prod_maps = set(re.findall('\:([[0-9]+)\]', prod))
    reactants_smi_list = []
    for mol in reactants:
        if mol is None:
            continue
        used = False
        for a in mol.GetAtoms():
            if a.HasProp('molAtomMapNumber'):
                if a.GetProp('molAtomMapNumber') in prod_maps:
                    used = True
                else:
                    a.ClearProp('molAtomMapNumber')
        if used:
            reactants_smi_list.append(Chem.MolToSmiles(mol, True))

    return reactants_smi_list


def sp(reaction):
    tmp = reaction.split('>')
    re_lst = []
    for smiles in tmp[0].split('.'):
        output = simple_canonicalize(smiles)
        if output is None:
            return None
        re_lst.append(output)

    output = simple_canonicalize(tmp[-1])
    if output is None:
        return None
    product = output

    reactants = ".".join(sorted(get_contrib_reactants_smiles(product, re_lst)))
    return (product, reactants)


def am(data):
    new_dict = {}
    for p, reactants_dict in data.items():
        mol = Chem.MolFromSmiles(p)
        [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        key = Chem.MolToSmiles(mol)
        key = simple_canonicalize(key)
        
        tmp_lst = []
        for reactants, id_strs in reactants_dict.items():
            tmp = []
            for r in reactants.split('.'):
                mol = Chem.MolFromSmiles(r)
                [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
                sml = Chem.MolToSmiles(mol)
                sml = simple_canonicalize(sml)
                tmp.append(sml)
            tmp_str = '.'.join(sorted(tmp))
            if tmp_str != '':
                tmp_lst.append((tmp_str, id_strs))
        if len(tmp_lst) != 0:
            if key not in new_dict:
                new_dict[key] = tmp_lst
            else:
                new_tmp, tmp_lst = [], tmp_lst + new_dict[key]
                for _ in tmp_lst:
                    if _ not in new_tmp:
                        new_tmp.append(_)
                new_dict[key] = new_tmp
    return new_dict

