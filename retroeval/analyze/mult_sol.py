import itertools
import pickle
from collections import defaultdict, Counter

from retroeval.utils.data import load_multisol_dataset
from retroeval.utils.mol import canonicalize
from retroeval.utils.file import dict_to_plt
from retroeval.utils.config import get_default_ss_results_dir


def multi_sol_analysis(models, dataset, results_dir, atks=[3, 5, 10, 50]):
    for k in atks:
        assert k <= 50  # we assume  this below

    prods, reacts = load_multisol_dataset(dataset, canonical=True)

    sols = list(itertools.chain.from_iterable([range(1, len(sols)+1) for sols in reacts]))
    t_sol_cts = dict(Counter(sols))

    for model in models:
        with open(f'{results_dir}/{model}_pred_lst.pkl', 'rb') as f:
            preds = pickle.load(f)

        assert len(preds) == len(prods)

        perk_p_sol_cts = {k: defaultdict(int) for k in atks}

        for i, pred in enumerate(preds):
            ts = reacts[i]  # prediction targets
            ps = [canonicalize(ri) for ri in pred['reactants'][:50]]
            for k in atks:
                nfound = 0
                for t in ts:
                    if t in ps[:k]:
                        nfound += 1
                        perk_p_sol_cts[k][nfound] += 1
        for k in atks:
            print(k)
            p_sol_cts = perk_p_sol_cts[k]
            print(p_sol_cts)

            # for all numbers of solutions we have, average how many tasks we solved
            # eg at k=3 we may have 10 tasks (=products) that have 2 different solutions,
            # so t_sol_cts[2] = 10, p_sol_cts[2] = the actual number of products where we found 2 solutions
            for n in t_sol_cts:
                p_sol_cts[n] = p_sol_cts[n]/t_sol_cts[n]

            print(dict_to_plt(p_sol_cts, f"./{results_dir}/eval/{dataset}_{model}_nsol_k{k}.txt"))


if __name__ == "__main__":

    models = ["mlp", "graphretro"]
    data = "uspto-ms"
    checkpoint = "uspto-50k"
    results_dir = get_default_ss_results_dir(data, "test", checkpoint)
    multi_sol_analysis(models, data, results_dir)



