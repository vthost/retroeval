import argparse
import os
import pickle
from retroeval.model.factory import create_single_step_model
from retroeval.utils.data import load_dataset
from retroeval.utils.config import get_default_ss_results_dir


def run_ss(model, checkpoint, data, part, batch_size, results_dir):
    kwargs = {
        "use_prob_thr": False,  # not used currently
        "topk": 50
    }

    m = create_single_step_model(model, checkpoint=checkpoint, **kwargs)
    prods, _ = load_dataset(data, part=part)
    print("Successfully loaded")

    pred_lst = []
    curr_idx, batch_size = 0, batch_size
    while curr_idx < len(prods):
        pred_lst += m.run_all(prods[curr_idx:curr_idx + batch_size])
        curr_idx = curr_idx + batch_size
        # break
        # print(b)

    os.makedirs(results_dir, exist_ok=True)

    with open(f'{results_dir}/{model}_pred_lst.pkl', 'wb') as f:
        pickle.dump(pred_lst, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mlp")
    parser.add_argument('--checkpoint', type=str, default="uspto-50k")
    parser.add_argument('--data', type=str, default="uspto-50k")
    parser.add_argument('--part', type=str, default="test")
    parser.add_argument('--results_dir', type=str, default="")
    # only used to create default results_dir if that is not provided, if not provided we use checkpoint
    parser.add_argument('--exp_id', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('-f_atoms', type=str, default="")  # this is a dummy, otherwise GLN won't work
    args = parser.parse_args()

    exp_id = args.exp_id if args.exp_id else args.checkpoint
    results_dir = args.results_dir if args.results_dir else get_default_ss_results_dir(args.data, args.part, exp_id)

    run_ss(args.model, args.checkpoint, args.data, args.part, args.batch_size, results_dir)

