from collections import defaultdict
from rdchiral import main
from rdchiral.initialization import rdchiralReaction, rdchiralReactants

from retroeval.utils.config import DATASET_INFO


# GLN's Reactor adapted
class _TemplateRunner(object):

    def __init__(self):
        self.rxn_prep = {}
        self.smi_prep = {}
        self.cached_results = {}

    def _get_rdchiral_rxn(self, rxn):
        p, a, r = rxn.split('>')
        if '.' in p:  # we assume the product has only one molecule
            if p[0] != '(':
                p = '('+p+')'
        rxn = '>'.join((p, a, r))
        if not rxn in self.rxn_prep:
            try:
                t = rdchiralReaction(rxn)
            except:
                t = None
            self.rxn_prep[rxn] = t
        return self.rxn_prep[rxn]

    def _get_rdchiral_reacts(self, smiles):
        if not smiles in self.smi_prep:
            self.smi_prep[smiles] = rdchiralReactants(smiles)
        return self.smi_prep[smiles]

    def run(self, smiles, template):
        key = (smiles, template)
        if key in self.cached_results:
            return self.cached_results[key]
        rxn = self._get_rdchiral_rxn(template)
        src = self._get_rdchiral_reacts(smiles)
        if rxn is None or src is None:
            return None
        try:
            outcomes = main.rdchiralRun(rxn, src)
            self.cached_results[key] = outcomes
        except:
            self.cached_results[key] = None
        return self.cached_results[key]


TemplateRunner = _TemplateRunner()


# Note we here consider a multi-solution format in task, i.e., all_reacts is a list of possibly multiple solutions
def match_templates(tpl_smarts_list, task):
    prod, all_reacts = task
    print(prod)

    tpl_succ = defaultdict(lambda: defaultdict(list))
    tbd = [reacts for reacts in all_reacts]

    for i, tpl in enumerate(tpl_smarts_list):
        res = TemplateRunner.run(prod, tpl)
        if res is None or not res:
            continue

        for reacts in all_reacts:
            if reacts in res:
                tpl_succ[prod][reacts] += [i]
                if reacts in tbd:
                    tbd.remove(reacts)

    n_found, n_tbd = len(reacts) - len(tbd), len(tbd)
    n_found1 = 1 if n_found > 0 else 0  # True
    return n_found1, n_found, n_tbd, tpl_succ


if __name__ == "__main__":
    p = "COC(=O)Cc1ccc(C#Cc2cc(C(C)(C)C)c(OC(C)C)c(CBr)c2C)cc1"
    r = "COC(=O)Cc1ccc(C#Cc2cc(C(C)(C)C)c(OC(C)C)c(CO)c2C)cc1.O=C1CCC(=O)N1Br"
    s = "O=C1CCC(=O)N1[Br:1].O[CH2:2][c:3]1[c:4]([CH3:5])[c:6]([C:7]#[C:8][c:9]2[cH:10][cH:11][c:12]([CH2:13][C:14]([O:15][CH3:16])=[O:17])[cH:18][cH:19]2)[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[O:27][CH:28]([CH3:29])[CH3:30]>>[Br:1][CH2:2][c:3]1[c:4]([CH3:5])[c:6]([C:7]#[C:8][c:9]2[cH:10][cH:11][c:12]([CH2:13][C:14]([O:15][CH3:16])=[O:17])[cH:18][cH:19]2)[cH:20][c:21]([C:22]([CH3:23])([CH3:24])[CH3:25])[c:26]1[O:27][CH:28]([CH3:29])[CH3:30]"
    t = "([Br;H0;D1;+0:3]-[CH2;D2;+0:1]-[c:2])>>(O-[CH2;D2;+0:1]-[c:2]).(O=C1-C-C-C(=O)-N-1-[Br;H0;D1;+0:3])"

    r = match_templates([t], (p, [r]))
    print(r)

    p = "Cc1ccc(C2=NNC(=O)CC2)cc1OCC1CO1"
    r = "Cc1ccc(C2=NNC(=O)CC2)cc1O.ClCC1CO1"
    t = "[#8:3]-[C:2]-[CH2;D2;+0:1]-[O;H0;D2;+0:4]-[c:5]>>Cl-[CH2;D2;+0:1]-[C:2]-[#8:3].[OH;D1;+0:4]-[c:5]"

    r = match_templates([t], (p, [r]))
    print(r)
    pass

