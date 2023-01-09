import argparse
import itertools
import os
import pickle
import pandas as pd
import numpy as np

from retroeval.utils.data import load_multisol_dataset
from retroeval.utils.mol import canonicalize
from retroeval.utils.file import df_to_plot, df_to_files
from retroeval.utils.config import MODEL_TO_PLOT, METRICS_ROUTE, get_default_ss_results_dir
from retroeval.utils.metrics_multisol import *  # used dynamically


def eval_ss_multisol(model, t_reacts_lst, evaluators, ks, max_k, results_dir):

    with open(f'{results_dir}/{model}_pred_lst.pkl', 'rb') as f:
        p_react_dicts = pickle.load(f)

    # assert len(t_reacts) == len(p_dicts)

    results = []
    for i, t_reacts in enumerate(t_reacts_lst):

        p_reacts = list(map(canonicalize, p_react_dicts[i]['reactants'][:max_k]))

        for e in evaluators:
            results += [e(t_reacts, p_reacts, ks=ks)]

    results = np.concatenate(results).reshape(len(t_reacts_lst), -1).mean(0)
    return results


def eval_ss_multisol_all(models, data, part, results_dir, metrics=["rec_k"], ks=[1, 5, 10], max_k=50,
             out=["csv", "latex"], plot=[]):

    # regular metrics
    all_results_df = pd.DataFrame()
    evaluators = [globals().get(f"{m}") for m in metrics if m not in METRICS_ROUTE]
    if evaluators:
        _, t_reacts = load_multisol_dataset(data, canonical=True)

        for model in models:

            results = eval_ss_multisol(model, t_reacts, evaluators, ks, max_k, results_dir)

            all_results_df = all_results_df.append(pd.DataFrame(results).T)

    # recording
    cols = [[f"{m[:-2]}-{k}" for k in ks] if m.endswith('_k') else [m] for m in metrics]
    all_results_df.columns = itertools.chain.from_iterable(cols)
    all_results_df.insert(loc=0, column='Model', value=models)

    path = f"{results_dir}/eval"
    if out:
        os.makedirs(path, exist_ok=True)
        df_to_files(all_results_df, f"{path}/{data}_tab", out)

    if plot:
        os.makedirs(path, exist_ok=True)
        all_results_df["Model"] = all_results_df["Model"].apply(lambda model: MODEL_TO_PLOT.get(model, model))
        for metric in plot:
            df_to_plot(all_results_df, "Model", metric, path=f"{path}/{data}_{metric}_plt.tex")

    print(all_results_df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default="mlp")  # comma separated str
    parser.add_argument('--data', type=str, default="uspto-ms")
    parser.add_argument('--part', type=str, default="test")
    parser.add_argument('--results_dir', type=str, default="")
    # only used to locate default results_dir if that is not provided
    parser.add_argument('--exp_id', type=str, default="uspto-50k")
    parser.add_argument('--metrics', type=str, default="rec_k")
    parser.add_argument('--ks', type=str, default="1,3,5,10")
    parser.add_argument('--out', type=str, default="csv")
    parser.add_argument('--plot', type=str, default="")
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else get_default_ss_results_dir(args.data, args.part, args.exp_id)

    eval_ss_multisol_all(args.models.split(","), args.data, args.part, results_dir,
                metrics=args.metrics.split(","), ks=[int(k) for k in args.ks.split(",")],
                out=args.out.split(",") if args.out else [], plot=args.plot.split(",") if args.plot else [])



