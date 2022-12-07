import pickle
import argparse
import pandas as pd
from utils.mol import canonicalize

from retroeval.utils.config import get_default_ss_results_dir
from retroeval.utils.mss import mss


def eval_mss_all(models, dataset, part, topk, results_dir):

    t_df = pd.read_csv(f'./data/{dataset}/{dataset}-{part}-rxns.txt')
    t_df['reactants'] = t_df['reactants'].apply(canonicalize)

    all_mss = {}
    for model in models:
        with open(f'{results_dir}/{model}_pred_lst.pkl', 'rb') as f:
            p_react_dicts = pickle.load(f)

        p_reacts = [list(map(canonicalize, p_react_dict['reactants'][:topk])) for p_react_dict in p_react_dicts]

        all_mss[model] = mss(t_df, p_reacts, topk=topk)  # this topk will not have any effect

    print(all_mss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default="mlp")  # comma separated str
    parser.add_argument('--data', type=str, default="rt-1k")
    parser.add_argument('--part', type=str, default="test")
    parser.add_argument('--results_dir', type=str, default="")
    # only used to locate default results_dir if that is not provided
    parser.add_argument('--exp_id', type=str, default="uspto-50k")
    parser.add_argument('--topk', type=int, default="10")
    parser.add_argument('--out', type=str, default="csv")
    parser.add_argument('--plot', type=str, default="")
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else get_default_ss_results_dir(args.data, args.part, args.exp_id)

    eval_mss_all(args.models.split(","), args.data, args.part, args.topk, results_dir)





