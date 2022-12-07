import os
import argparse
import pandas as pd
import traceback
from PARoutes.analysis.route_quality import main

from retroeval.utils.config import get_default_ms_results_dir
from retroeval.utils.file import df_to_files


def run_paroutes(methods, dataset, part, results_dir):
    repo = os.path.abspath("../")
    for method in methods:
        path = os.path.abspath(f"../{results_dir}/{method}")
        try:
         main(optional_args=[
            "--routes",
            f"{path}/routes.json",
            "--references",
            f"{repo}/data/{dataset}/{dataset}-{part}-routes.json",
            "--output",
            f"{path}/route_analyses.csv"])
        except Exception as e:
         print("Error with ", method)
         traceback.print_exc()


def summarize_paroutes_analysis(methods, results_dir, out=""):
    # NOTE: paroutes output columns (to select) are as follows:
    # ref lrr,ref nleaves,solved target,
    # max llr-1,min llr-1,min nleaves-1,max leaves overlap-1,mean solved-1,best-1,
    # max llr-5,min llr-5,min nleaves-5,max leaves overlap-5,mean solved-5,best-5,
    # max llr-10,min llr-10,min nleaves-10,max leaves overlap-10,mean solved-10,best-10,
    # true_rank
    cols = ["solved target", "true_rank", "max leaves overlap-1", "max leaves overlap-5", "max leaves overlap-10"]
    all_results = []

    for method in methods:
        p = f"{results_dir}/{method}/route_analyses.csv"
        if not os.path.exists(p): continue

        # if you want to focus on single-step, uncomment this to ignore search_algo in output table
        # method = method.split('_')[0]

        data = pd.read_csv(p, usecols=cols)
        ld = len(data)
        result = {
            "Model": method,
            # "solved": len(data[data['solved target'] == True]) / ld,  # not as indicative for non-template-based model
            "top-1": len(data[data['true_rank'] == 1]) / ld,
            "top-5": len(data[data['true_rank'] <= 5]) / ld,
            "top-10": len(data[data['true_rank'] <= 10]) / ld,
            "mlo-1": sum(data['max leaves overlap-1']) / ld,
            "mlo-5": sum(data['max leaves overlap-5']) / ld
        }
        all_results += [result]

    df = pd.DataFrame.from_records(all_results)
    df_to_files(df, f"{results_dir}/results_tab", out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # comma separated str of '${model}-${search_algo}', all in results_dir if empty
    parser.add_argument('--methods', type=str, default="mlp.uspto-50k_rs")
    parser.add_argument('--data', type=str, default="rt-tpl-100")
    parser.add_argument('--part', type=str, default="test")
    parser.add_argument('--results_dir', type=str, default="")  # leave empty for use of default code
    # only used to locate default results_dir if that is not provided (we use the checkpoint of the single-step model)
    parser.add_argument('--exp_id', type=str, default="uspto-50k")
    parser.add_argument('--out', type=str, default="csv")

    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir else get_default_ms_results_dir(args.data, args.part, args.exp_id)
    methods = args.methods.split(",") if args.methods else os.listdir(results_dir)

    run_paroutes(methods, args.data, args.part, results_dir)
    summarize_paroutes_analysis(methods, results_dir, args.out)
