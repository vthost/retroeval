import os
import pandas as pd
from collections import Counter

from retroeval.utils.route import load_routes
from retroeval.utils.file import dict_to_plt, df_to_files

def get_rxn_stats(dataset):
    f = f"./data/{dataset}/{dataset}-rxns.csv"
    if not os.path.exists(f):
        return

    df = pd.read_csv(f)
    stats = {}
    stats["nroutes"] = len(df["route_id"].drop_duplicates())
    stats["nreact"] = len(df)
    stats["npatents"] = len(df["id"].apply(lambda s: s.split(";")[0]).drop_duplicates())
    return stats


def record_route_stats(dataset, path=""):
    f = f"./data/{dataset}/{dataset}-routes.pickle"
    if not os.path.exists(f):
        return

    routes = load_routes(f)

    properties = ["nreactions", "nleaves", "nmols", "llr"]
    values = map(lambda r: [r[p] for p in properties], routes)

    values = list(zip(*values))  # separate, one tuple per property
    for i, property in enumerate(properties):
        val_dict = dict(Counter((values[i])))
        for k, v in val_dict.items():
            val_dict[k] = val_dict[k]/len(routes)

        print(val_dict)
        if not path:
            path = f"./data/{dataset}/stats/"
        print(dict_to_plt(val_dict, path=path + f"/{property}"))


if __name__ == "__main__":

    stats = {}
    for dataset in os.listdir("./data"):
        stats[dataset] = get_rxn_stats(dataset)

    df = pd.DataFrame.from_records(stats)
    df_to_files(df, file_name=f"./data/{dataset}/stats/rxn")

    record_route_stats()



