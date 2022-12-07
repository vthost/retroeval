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
    parser.add_argument('--dataset', type=str, default="uspto-50k")
    parser.add_argument('--multi_sol', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default="./assets/templates/")
    args = parser.parse_args()

    #########################################################
    os.makedirs(args.out_dir, exist_ok=True)

    if args.multi_sol:
        prods, reacts = load_multisol_dataset(args.data)
    else:
        prods, reacts = load_dataset(args.data)
        ps, rs = [], []

        for i, p in enumerate(prods):
            if p not in ps:
                ps += [p]
                rs += [[rs[i]]]
            else:         # merge duplicates
                j = ps.index(p)
                rs[j] += [rs[i]]
        prods, reacts = ps, rs

    # to test
    # prods = ["Cc1ccc(C2=NNC(=O)CC2)cc1OCC1CO1"]
    # reacts = [["Cc1ccc(C2=NNC(=O)CC2)cc1O.ClCC1CO1"]]
    # templates = ["[#8:3]-[C:2]-[CH2;D2;+0:1]-[O;H0;D2;+0:4]-[c:5]>>Cl-[CH2;D2;+0:1]-[C:2]-[#8:3].[OH;D1;+0:4]-[c:5]"]

    nums = sum(map(len, reacts))
    templates = load_templates(args.templates, parts=["train", "valid"])

    match_templates_init = partial(match_templates, templates)  # cannot use lambda in multiprocessing

    template_succs = {}
    ct, tbdct, min1ct = 0, 0, 0

    njobs = int(multiprocessing.cpu_count())
    with Pool(njobs) as pool:
        results = pool.map(match_templates_init, zip(prods, reacts))

    # min1succ = found a template for one of the possible solutions,
    # ok = found templates for all possible solutions
    # tbd = the ones not found
    min1succ, succ, tbd, template_succs1 = list(zip(*results))
    min1succ, succct, tbdct = sum(min1succ), sum(succ), sum(tbd)

    for dic in template_succs1:
        template_succs.update(dic)

    numt = len(prods)
    nums = sum(map(len, reacts))
    print("*"*20)
    print(f"abs: {min1ct}/{numt-min1ct} , {succct}/{tbdct}")  # should be same for regular data with only one solution
    print(f"perc: {min1ct/numt:.4f}/{(numt-min1ct)/numt:.4f} , {succct/nums:.4f}/{tbdct/nums:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(template_succs, f"{args.outdir}/d_{args.dataset}_tpl_{args.templates}_succ.pt")
