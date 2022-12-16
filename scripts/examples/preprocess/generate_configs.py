import json
import argparse
import itertools
import os
from itertools import chain, combinations

from examples.utils.mol import FP


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))


vary = {
   "fp_types": map(list, powerset(FP.values()))
   # "fp_dim": [2048, 4096],
   # "hidden_dim": [2048, 4096],
   # "layers": [2, 3],
   # "dropout": [0, 0.3, 0.5]
}


base = {
    "fp_radii": 2,
    "fp_dim": 2048,
    "hidden_dim": 4096,
    "out_dim": 0,
    "layers": 2,
    "dropout": 0.3,
    "batch_norm": True,
    "activation": "elu"
}


def generate_configs(path):
    c = 0
    for config in itertools.product(*vary.values()):
        config = {k: config[i] for i, k in enumerate(vary.keys())}
        print(config)
        config.update(base)
        json.dump(config, open(f"{path}/config_{c}.json", "w"), indent=2)
        c += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='"./examples/configs/"',
                        type=str)
    args = parser.parse_args()

    os.makedirs(args.directory, exist_ok=True)
    generate_configs(args.directory)
