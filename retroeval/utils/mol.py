import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        tmp = Chem.RemoveHs(tmp)
    except:
        print("Exception;", smiles)
        return smiles
    if tmp is None:
        print("None;", smiles)
        return smiles
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)


# allow for some reactant failures by dropping them
# return None if failure
def canonicalize_all(smiles):
    if '.' in smiles:
        new_lst, lst = [], smiles.split('.')
        for smiles_i in lst:
            smiles_i = canonicalize_all(smiles_i)
            if not smiles_i or smiles_i is None:
                continue
            new_lst.append(smiles_i)
        smiles = '.'.join(sorted(new_lst)) if new_lst else None
    else:
        try:
            m = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(m)
        except Exception:
            return None
    return smiles