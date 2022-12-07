from scripts.paroutes.select_routes_time import save_data, _get_args
from retroeval.utils.route import *


if __name__ == '__main__':

    dataset = "rd"
    p0 = f"./non_overlapping_routes_sorted.pickle"
    p1 = f"./data/{dataset}/{dataset}-valid-routes.pickle"
    p2 = f"./data/{dataset}/{dataset}-test-routes.pickle"
    # for the rt data we need to see the printed output from the test data creation
    # rt: Stopped test data extraction at next_idx ...
    rt_next_id = 0

    all = load_routes(p0)
    valid = load_routes(p1)
    test = load_routes(p2)

    if rt_next_id:  # shortcut for rt since we know which patents to skip
        all = all[rt_next_id:]
        test_patent_ids = []
    else:
        test_patent_ids = list(set([r['id'].split('@')[0] for r in test]))

    # for rt we wouldn't need test since all doesn't contain them
    valid_test_route_ids = [r['id'] for r in valid + test]

    train = []
    for route in all:
        if route['id'] not in valid_test_route_ids and route['id'].split('@')[0] not in test_patent_ids:
            train += [route]

    print(f"Extracted {len(train)} train routes")

    args = _get_args()
    save_data(train, args, dataset + "-train")

