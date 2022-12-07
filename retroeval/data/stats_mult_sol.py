import itertools
from collections import defaultdict, Counter

from retroeval.utils.data import load_multisol_dataset


def calc_multi_sol_stats(dataset):
    dataset_dict = load_multisol_dataset(dataset, return_dict=True)

    print("# tasks (=products)", len(dataset_dict))

    # histogram
    h = defaultdict(int)
    for v in dataset_dict.values():
        h[len(v)] += 1

    print("# solutions # tasks")
    ct = 0
    for k, v in h.items():
        print(f"{k}\t{v}")
        ct += k*v

    print("# solutions # tasks - cumulative")
    sols = list(itertools.chain.from_iterable([range(1, len(sols)+1) for sols in dataset_dict.values()]))
    t_sol_cts = dict(Counter(sols))
    print(t_sol_cts)

    print("# solutions overall", ct)


if __name__ == "__main__":

    calc_multi_sol_stats("uspto-ms")



