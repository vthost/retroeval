# check which templates of given "templates" dataset are applicable/successful for the tasks in "dataset"

import argparse
import os
import torch
from functools import partial
import multiprocessing
from multiprocessing import Pool

from retroeval.utils.data import load_dataset, load_multisol_dataset, load_templates
from retroeval.utils.template import match_templates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--templates', type=str, default="uspto-50k")
    parser.add_argument('--data', type=str, default="uspto-50k")
    parser.add_argument('--multi_sol', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default="./assets/templates/")
    args = parser.parse_args()

    #########################################################
    os.makedirs(args.out_dir, exist_ok=True)

    if args.multi_sol:
        prods, reacts = load_multisol_dataset(args.data, canonical=True)
    else:
        prods, reacts = load_dataset(args.data, canonical=True, part='test')
        ps, rs = [], []

        for i, p in enumerate(prods):
            if p not in ps:
                ps += [p]
                rs += [[reacts[i]]]
            else:         # merge duplicates
                j = ps.index(p)
                rs[j] += [reacts[i]]
        prods, reacts = ps, rs

    # dummy data to test
    # prods = ["Cc1ccc(C2=NNC(=O)CC2)cc1OCC1CO1", "Cc1ccc(C2=NNC(=O)CC2)cc1OCC1CO1"]
    # reacts = [["Cc1ccc(C2=NNC(=O)CC2)cc1O.ClCC1CO1", "dummy"], ["Cc1ccc(C2=NNC(=O)CC2)cc1O.ClCC1CO1", "dummy", "dummy2"]]
    # templates = ["[#8:3]-[C:2]-[CH2;D2;+0:1]-[O;H0;D2;+0:4]-[c:5]>>Cl-[CH2;D2;+0:1]-[C:2]-[#8:3].[OH;D1;+0:4]-[c:5]"]

    nums = sum(map(len, reacts))
    templates = load_templates(args.templates, parts=["train", "valid"])

    match_templates_init = partial(match_templates, templates)  # cannot use lambda in multiprocessing

    njobs = int(multiprocessing.cpu_count())
    with Pool(njobs) as pool:
        results = pool.map(match_templates_init, zip(prods, reacts))
    # results = [match_templates_init(t) for t in zip(prods, reacts)]
    # min1: count of products for which we found a template for one of the possible solutions
    # succ: count of products for which we found templates for all possible solutions
    # tbd: count of reactant sets for which no template was found
    min1ct, succct, tbdct, template_succs1 = list(zip(*results))
    min1ct, succct, tbdct = sum(min1ct), sum(succct), sum(tbdct)

    template_succs = {}
    for dic in template_succs1:
        template_succs.update(dic)

    num_tasks = len(prods)
    nums = sum(map(len, reacts))
    print("*"*20)
    print("solved 1/unsolved, solved/unsolved:")
    print(f"abs:\t {min1ct}/{num_tasks-min1ct}, {succct}/{tbdct}")  # should be same for regular data with only one solution
    print(f"perc:\t {min1ct/num_tasks:.4f}/{(num_tasks-min1ct)/num_tasks:.4f}, {succct/nums:.4f}/{tbdct/nums:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(template_succs, f"{args.out_dir}/d_{args.data}_tpl_{args.templates}_succ.pt")


