import argparse
import itertools
import pandas as pd
from retroeval.utils.route import load_routes, map_rxn_nodes
from retroeval.utils.mol import canonicalize


def extract_metadata(rxn_node, rlevel, **kwargs):
    metadata = rxn_node["metadata"]
    metadata["level"] = rlevel
    if 'ID' in metadata:
        metadata["id"] = metadata["ID"]
        del metadata["ID"]
    else:
        assert 'id' in metadata
    return metadata


def extract_reactions(dataset, part):
    routes_file = f"./data/{dataset}/{dataset}-{part}-routes.pickle"
    rxns_file = f"./data/{dataset}/{dataset}-{part}-rxns.txt"

    # list of dictionaries, one per route/target
    routes = load_routes(routes_file)

    # list of list of dictionaries, one list per route, one dict per reaction
    rxn_dicts = [map_rxn_nodes(r['rt'], extract_metadata) for r in routes]
    for i, rxn_dicts2 in enumerate(rxn_dicts):
        for rxn_dict in rxn_dicts2:
            rxn_dict['route_id'] = routes[i]['id']

    rxn_dicts = list(itertools.chain.from_iterable(rxn_dicts))
    df = pd.DataFrame.from_records(rxn_dicts, columns=['id', 'smiles', 'level', 'patent_id',
                                                       'route_id', 'retro_template', "template_hash"])
    df["product"] = df["smiles"].apply(lambda s: s.split(">"))
    df["reactants"] = df["product"].apply(lambda s: canonicalize(s[0]))
    df["product"] = df["product"].apply(lambda s: canonicalize(s[-1]))
    df['patent_id'] = df['id'].apply(lambda rxnid: rxnid.split(';')[0])
    df.to_csv(rxns_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--part', type=str, default="")
    args = parser.parse_args()

    extract_reactions(args.dataset, args.part)


