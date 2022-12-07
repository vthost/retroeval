import random

from scripts.paroutes.select_routes_time import _get_args, save_data
from retroeval.utils.mol import canonicalize
from retroeval.utils.route import load_routes, check_route, extract_rxn_smiles
from retroeval.utils.data import load_dataset

random.seed(3)


def no_node_scores_route_check(node_score, _):
    if node_score > 0:
        return 0  # some rxns score
    return 1


# true if there is overlap at node
if __name__ == '__main__':

    data = "rd"
    size = 1000
    overlap_data = "uspto-50k"
    overlap_data = load_dataset(overlap_data, part="train")[0] + load_dataset(overlap_data, part="valid")[0]

    # true if in that data
    def node_check_overlap(node):
        rxn_smi = node['metadata']['smiles'].split(">")
        prod = canonicalize(rxn_smi[-1])
        return int(prod in overlap_data)

    def route_check_no_overlap(route):
        route_score, _, _ = check_route(node_check_overlap, no_node_scores_route_check,
                                        route['rt'] if 'rt' in route else route)
        return route_score

    p = f"./data/{data}/{data}-test-routes.pickle"
    args = _get_args()  # only for file name templates
    routes = load_routes(p)
    all_routes = []
    all_rxns = []

    i = 0
    patent_ids = set()
    idx = list(range(len(routes)))
    random.shuffle(idx)
    while len(all_routes) < size:
        next_route = routes[idx[i]]
        i += 1
        pid = next_route["id"].split("@")[0]
        if pid not in patent_ids:
            if route_check_no_overlap(next_route):
                # next check for reaction duplicates
                ok = 1
                rxns = extract_rxn_smiles(next_route)
                for rxn in rxns:
                    if rxn in all_rxns:
                        ok = 0
                        break
                if ok:
                    all_routes += [next_route]
                    all_rxns += rxns
                    patent_ids.add(pid)

    save_data(all_routes, args, f"{data}-{size//1000}k-test")




