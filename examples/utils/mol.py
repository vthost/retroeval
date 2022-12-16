import torch
from enum import Enum
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Avalon.pyAvalonTools


# https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-fingerprints
# https://www.rdkit.org/UGM/2012/Landrum_RDKit_UGM.Fingerprints.Final.pptx.pdf
# https://www.rdkit.org/docs/RDKit_Book.html#additional-information-about-the-fingerprints
class FP(Enum):
    ATOMPAIR = "atompair"
    AVALON = "avalon"
    ECFP = "ecfp"
    ERG = "erg"
    LAYERED = "layered"
    MAACS = "maacs"
    MORGAN = "morgan"
    PATTERN = "pattern"
    RDK = "rdk"
    TOPOLOGICAL_TORSION = "topological_torsion"

    @staticmethod
    def values():
        return [fp.value for fp in FP]


def calc_fp_dim(fp_dim, fp_types):
    return fp_dim * (len(fp_types) - 1) + 167 if FP.MAACS in fp_types or FP.MAACS.value in fp_types else fp_dim * len(fp_types)


def fp_dict_to_np(fp_dict, dim):
    fp = np.zeros(dim)
    for k, v in fp_dict.items():
        fp[k % dim] += v  # Note modulo may yield to merged values
    return fp


# https://github.com/rdkit/rdkit/discussions/3863
# or np.frombuffer(bytes(ebv.ToBitString(), 'utf-8'), 'u1') - ord('0')
def fp_bitvec_to_np(fp_bitvec):
    return np.frombuffer(fp_bitvec.ToBitString().encode(), 'u1') - ord('0')


def mol_to_fp(mol, fp_type, dim=2048, radius=2):
    if isinstance(fp_type, str):
        fp_type = FP(fp_type)

    if fp_type == FP.ATOMPAIR:
        fp = AllChem.GetAtomPairFingerprint(mol).GetNonzeroElements()
    elif fp_type == FP.AVALON:
        fp = rdkit.Avalon.pyAvalonTools.GetAvalonFP(mol, nBits=dim)
    elif fp_type == FP.ECFP:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=dim, useChirality=True, useFeatures=False)
    elif fp_type == FP.ERG:
        fp = AllChem.GetErGFingerprint(mol)
        fp = {i: fp[i] for i in np.nonzero(fp)[0]}
    elif fp_type == FP.LAYERED:
        fp = AllChem.LayeredFingerprint(mol, fpSize=dim)
    elif fp_type == FP.MAACS:
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == FP.MORGAN:
        fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=True, useFeatures=True,
                                          useCounts=True).GetNonzeroElements()
    elif fp_type == FP.PATTERN:
        fp = Chem.PatternFingerprint(mol, fpSize=dim)
    elif fp_type == FP.RDK:
        fp = Chem.RDKFingerprint(mol, fpSize=dim)
    elif fp_type == FP.TOPOLOGICAL_TORSION:
        fp = AllChem.GetTopologicalTorsionFingerprint(mol).GetNonzeroElements()
    else:
        raise ValueError("No valid fingerprint type provided.")

    if isinstance(fp, dict):
        fp = fp_dict_to_np(fp, dim)
    else:
        fp = fp_bitvec_to_np(fp)
    # print(fp_type, fp.size)
    return fp


def encode_smiles(smiles, fp_dim, fp_types=[FP.MORGAN], fp_radii=1):
    if fp_radii == []:
        fp_radii = [1]*len(fp_types)
    elif type(fp_radii) == int:
        fp_radii = [fp_radii]*len(fp_types)
    else:
        assert len(fp_types) == len(fp_radii)

    if type(smiles) == str:
        smiles = [smiles]

    fps = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            fps += [np.zeros(calc_fp_dim(fp_dim, fp_types))]
        fp = []
        for i, fp_type in enumerate(fp_types):
            try:
                fp += [mol_to_fp(mol, fp_type, dim=fp_dim, radius=fp_radii[i])]
            except:
                fp += [np.zeros(167) if fp_type == FP.MAACS else np.zeros(fp_dim)]
        fps += [np.concatenate(fp)]

    fps = np.stack(fps)
    return torch.from_numpy(fps).float()


if __name__ == "__main__":
    fpd = calc_fp_dim(20, fp_types=FP.values())

    smi = "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC"
    mol = Chem.MolFromSmiles(smi)
    for fpt in FP:
        print(fpt.name, fpt.value)
        fp = mol_to_fp(mol, fpt)
        print(fp)
        assert np.sum(fp) > 0

    encode_smiles(smi, 2048, fp_types=FP.values())
