import gzip
import json
import pickle
from functools import partial


def load_routes(filename):
    routes_list = []
    if filename.endswith("json.tar.gz"):
        with gzip.open(filename, "rt", encoding="UTF-8") as fileobj:
            routes = json.load(fileobj)

    elif filename.endswith(".pickle"):
        with open(filename, "rb") as fileobj:
            routes = pickle.load(fileobj)
    else:
        with open(filename) as fileobj:
            routes = json.load(fileobj)
    routes_list.extend(routes)

    return routes_list


def extract_rxn_smiles(route):
    if 'rt' in route:
        route = route['rt']

    def extract_node_smiles(node, rlevel=0):
        return node['metadata']['smiles']
    return map_rxn_nodes(route, extract_node_smiles)


# func should given a (partial) route compute something on its root node
# (eg extract some data)
# considers only reaction nodes (ie no molecule nodes)
# it should accept kwargs (r-prefixed to avoid duplicates) so that we may add context in the future
# results of func will be collected in simple list
def map_rxn_nodes(route, func, level=-1):
    if route['type'] == 'reaction':
        results = [func(route, rlevel=level)]
        next_level = level
    else:  # mol node
        results = []
        next_level = level+1

    if 'children' in route:
        for c in route['children']:
            results += map_rxn_nodes(c, func, level=next_level)

    return results


# check each node of route using func, func must return a number per node
# Note: we only consider reaction nodes!
def _route_check(route, func):
    score, check_ct = 0, 0
    if route['type'] == 'reaction':
        score = func(route)
        check_ct = 1

    if 'children' in route:
        for c in route['children']:
            tmp_score, tmp_check_ct = _route_check(c, func)
            score += tmp_score
            check_ct += tmp_check_ct

    return score, check_ct


def check_route(func, rs_func, route):
    route_rxn_score, route_rxn_ct = _route_check(route, func)
    route_score = rs_func(route_rxn_score, route_rxn_ct)
    return route_score, route_rxn_score, route_rxn_ct


# check routes for one target
def _check_trg_routes(func, rs_func, routes):
    route_scores, rxn_score, rxn_ct = [], 0, 0
    for route in routes:
        route_score, route_rxn_score, route_rxn_ct = check_route(func, rs_func, route)
        rxn_score += route_rxn_score
        rxn_ct += route_rxn_ct
        route_scores += [route_score]
    return route_scores, rxn_score, rxn_ct


def check_routes(route_file, func, rs_func):
    all_routes = load_routes(route_file)
    _check_trg_routes_init = partial(_check_trg_routes, func, rs_func)

    import multiprocessing
    from multiprocessing import Pool
    njobs = int(multiprocessing.cpu_count())
    with Pool(njobs) as pool:
        results = pool.map(_check_trg_routes_init, all_routes) # list, one tuple per target's routes

    all_route_scores, rxn_scores, rxn_cts = list(zip(*results))
    return list(all_route_scores), sum(rxn_scores), sum(rxn_cts)


# returns true iff we get 1 for each rxn node (need to subtract root since no reaction for that)
def rscore_rxn_node_ct(node_score, rxn_node_ct):
    return int(node_score == rxn_node_ct)


# just to test with a check which gives each non-leaf node a 1.0 score
# returns 1 for every reaction node
def test_node_check(_):
    return 1


if __name__ == "__main__":
    # test: check_routes
    rp = "./results/sample/chemformer_rs/routes.json"
    all_route_scores, rxn_score, rxn_ct = check_routes(rp, test_node_check, rscore_rxn_node_ct)
    print(all_route_scores, rxn_score, rxn_ct)