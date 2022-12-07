import os
import json
import shutil
import argparse
from aizynthfinder.aizynthfinder import AiZynthFinder
import logging  #aizynthfinder.utils.logging as logging

from retroeval.utils.config import get_default_ms_results_dir


def init_files(results_dir, config_file, policy, finder):
    search = "rs" if "retrostar" in finder.config.search_algorithm else finder.config.search_algorithm
    save_dir = f"{results_dir}/{policy}_{search}"
    save_stats = f"{save_dir}/stats.jsonl"
    save_routes = f"{save_dir}/routes.json"

    # shutil.rmtree(save, ignore_errors=True)
    if not os.path.exists(save_dir):
        os.makedirs(f"{save_dir}")

    # If a previous run finished half we try to complete it now
    sidx = 0
    if os.path.exists(save_stats):
        with open(save_stats) as f:
            sidx = len(list(f.readlines()))
        with open(save_routes) as f:
            ls = list(f.readlines())
        if ls[-1].strip() != ",":
            with open(save_routes, "w") as f:
                f.write("\n".join(ls + [",\n"]))
    else:
        # our dummy json writer, in order to be able to use "a"
        with open(save_routes, "w") as f:
            f.write("[\n")

    # record this
    shutil.copyfile(config_file, f"{save_dir}/{config_file[config_file.rindex('/'):]}")

    return save_dir, save_stats, save_routes, sidx


def run_aizynthfinder(config_file, stock, policy, targets_file, results_dir):
    print("Here we go, we managed to enter main!")
    finder = AiZynthFinder(configfile=config_file)

    finder.stock.select(stock)
    finder.expansion_policy.select(policy)

    save_dir, save_stats, save_routes, sidx = init_files(results_dir, config_file, policy, finder)

    logger = finder._logger
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(f"{save_dir}/out.log"))

    with open(targets_file, 'r') as f:
        last = len(list(f.readlines())) - 1
    with open(targets_file, 'r') as f:
        for i, line in enumerate(f):
            if i < sidx: continue
            finder.target_smiles = line.strip()
            finder.prepare_tree()  # need to clear from last target
            finder.tree_search()

            finder.build_routes()
            stats = finder.extract_statistics()
            # logger.info(f"{i}: {stats}")

            with open(save_stats, "a") as f:
                f.write(json.dumps(stats) + '\n')
            with open(save_routes, "a") as f:
                f.write("[" + ", ".join(finder.routes.jsons) + "]\n")
                if i < last:
                    f.write(", \n")

    with open(save_routes, "a") as f:
        f.write("\n]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'./assets/aizynthfinder/config.yml')
    parser.add_argument('--stock', type=str, default='rt-tpl-100')
    parser.add_argument('--policy', type=str, default="mlp.uspto-50k")
    # below we use an id for a file ./data/rt-tpl-100/rt-tpl-100-test-targets.txt')
    # but a path to a target file works as well
    parser.add_argument('--data', type=str, default='rt-tpl-100')
    parser.add_argument('--part', type=str, default='test')
    parser.add_argument('--results_dir', type=str, default="")
    # only used to create default results_dir if that is not provided (we use the checkpoint of the single-step model)
    parser.add_argument('--exp_id', type=str, default="")  #
    parser.add_argument('-f_atoms', type=str, default="")  # this is a dummy, otherwise GLN won't work

    args = parser.parse_args()

    if os.path.isfile(args.data):
        targets_file = args.data
        # if args.data is a file path, use something like this
        targets_id = os.path.basename(args.data)[:os.path.basename(args.data).rindex['.']]\
            .replace("-test-targets", "").replace("-targets", "")      # for our data format remove the latter part
    else:
        targets_file = f"./data/{args.data}/{args.data}-{args.part}-targets.txt"
        targets_id = args.data

    exp_id = args.exp_id if args.exp_id else args.policy.split(".")[1]
    results_dir = args.results_dir if args.results_dir else get_default_ms_results_dir(args.data, args.part, exp_id)

    run_aizynthfinder(args.config, args.stock, args.policy, targets_file, results_dir)
