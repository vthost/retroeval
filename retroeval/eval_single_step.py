import argparse
import itertools
import os
import pickle
import pandas as pd
import numpy as np

from retroeval.utils.data import load_dataset
from retroeval.utils.mol import canonicalize
from retroeval.utils.metrics import *  # is used dynamically
from retroeval.utils.mss import mss
from retroeval.utils.file import df_to_plot, df_to_files
from retroeval.utils.config import MODEL_TO_PLOT, METRICS_ROUTE, get_default_ss_results_dir


def get_df_columns(metrics, ks, route=True):
    cols = [[f"{m[:-2]}-{k}" for k in ks] if m.endswith('_k') else [m] for m in metrics if m not in METRICS_ROUTE]
    if route:
        cols += [[f"{m[:-2]}-{k}" for k in ks] if m.endswith('_k') else [m] for m in metrics if m in METRICS_ROUTE]
    return itertools.chain.from_iterable(cols)


def eval_ss(model, t_reacts, evaluators, ks, max_k, results_dir):

    with open(f'{results_dir}/{model}_pred_lst.pkl', 'rb') as f:
        p_react_dicts = pickle.load(f)

    # assert len(t_reacts) == len(p_dicts)

    results = []
    for i, t_react in enumerate(t_reacts):

        p_reacts = list(map(canonicalize, p_react_dicts[i]['reactants'][:max_k]))

        for e in evaluators:
            results += [e(t_react, p_reacts, ks=ks)]
            # print(results)

    results = np.concatenate(results).reshape(len(t_reacts), -1).mean(0)
    return results


def eval_ss_route_metrics(model, t_df, evaluators, ks, max_k, results_dir):

    with open(f'{results_dir}/{model}_pred_lst.pkl', 'rb') as f:
        p_react_dicts = pickle.load(f)

    p_reacts = [list(map(canonicalize, p_react_dict['reactants'][:max_k])) for p_react_dict in p_react_dicts]

    results = []
    for e in evaluators:
        results += [e(t_df, p_reacts, ks=ks)]

    results = np.concatenate(results).reshape(1, -1)
    return results


def eval_ss_all(models, data, part, results_dir, metrics=["top_k", "mrr"], ks=[1, 5, 10], max_k=50,
             out=["csv", "latex"], plot=["top-1", "top-10"]):

    # regular metrics
    all_results_df = pd.DataFrame()
    evaluators = [globals().get(f"{m}") for m in metrics if m not in METRICS_ROUTE]
    if evaluators:
        _, t_reacts = load_dataset(data, part=part, canonical=True)

        for model in models:

            results = eval_ss(model, t_reacts, evaluators, ks, max_k, results_dir)

            all_results_df = all_results_df.append(pd.DataFrame(results).T)

    # route-context metrics
    path = f'./data/{data}/{data}-{part}-rxns.csv'
    evaluators = [globals().get(f"{m}") for m in metrics if m in METRICS_ROUTE]
    route = False  # default
    if evaluators and os.path.exists(path):
        t_df = pd.read_csv(path)
        if "route_id" in t_df.columns:
            route = True
            all_results_df2 = pd.DataFrame()
            t_df['reactants'] = t_df['reactants'].apply(canonicalize)

            for model in models:
                results = eval_ss_route_metrics(model, t_df, evaluators, ks, max_k, results_dir)

                all_results_df2 = all_results_df2.append(pd.DataFrame(results).T)
            all_results_df = pd.concat([all_results_df, all_results_df2], axis=1, join='inner')

    # recording
    all_results_df.columns = get_df_columns(metrics, ks, route=route)
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
    parser.add_argument('--data', type=str, default="rt-1k")
    parser.add_argument('--part', type=str, default="test")
    parser.add_argument('--results_dir', type=str, default="")
    # only used to locate default results_dir if that is not provided
    parser.add_argument('--exp_id', type=str, default="uspto-50k")
    parser.add_argument('--metrics', type=str, default="maxfrag_k,top_k,mrr,mss")
    parser.add_argument('--ks', type=str, default="1,5,10")
    parser.add_argument('--out', type=str, default="csv")
    parser.add_argument('--plot', type=str, default="")
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else get_default_ss_results_dir(args.data, args.part, args.exp_id)

    eval_ss_all(args.models.split(","), args.data, args.part, results_dir,
                metrics=args.metrics.split(","), ks=[int(k) for k in args.ks.split(",")],
                out=args.out.split(",") if args.out else [], plot=args.plot.split(",") if args.plot else [])



