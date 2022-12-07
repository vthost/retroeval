

# t_react: SMILES str, p_reacts: sorted list of SMILES str
def mrr(t_react, p_reacts, **kwargs):
    mrr = 0
    if t_react in p_reacts:
        mrr = 1 / (p_reacts.index(t_react) + 1)
    return [mrr]


def top_k(t_react, p_reacts, ks=[1, 5, 10]):
    ks = sorted(ks)
    top_ks = [0]*len(ks)
    for i, k in enumerate(ks):
        if t_react in p_reacts[ks[i-1] if i else 0:k]:
            top_ks[i] = 1
            if k < ks[-1]:
                for j, k2 in enumerate(ks[i+1:]):
                    top_ks[i+1+j] = 1
            break
    return top_ks


def maxfrag_k(t_react, p_reacts, ks=[1, 5, 10]):
    ks = sorted(ks)
    mf_ks = [0]*len(ks)

    t_react_ind = t_react.split(".")
    lens = [len(ri) for ri in t_react_ind]
    t_react_maxfrag = t_react_ind[lens.index(max(lens))]  # NOTE we take the first. one could check all.

    for i, k in enumerate(ks):
        p_reacts_frag = [r_pred.split(".") for r_pred in p_reacts[ks[i-1] if i else 0:k]]
        occurs = any([t_react_maxfrag in p_reacts_frag_j for p_reacts_frag_j in p_reacts_frag])
        if occurs:
            mf_ks[i] = 1
            if k < ks[-1]:
                for j, k2 in enumerate(ks[i+1:]):
                    mf_ks[i+1+j] = 1
            break
    return mf_ks


if __name__ == "__main__":
    all_p_reacts_reacts = [[1,2,3,4,5], [5,2,3,4,1], [6,2,3,4,5,1], [0,2,3,4,5,6]]

    t_react = 1
    for p_reacts in all_p_reacts_reacts:
        r = top_k(t_react, p_reacts)
        print(r)

    print()

    for p_reacts in all_p_reacts_reacts:
        r = mrr(t_react, p_reacts)
        print(r)

    print()

    t_react = "1"
    for p_reacts in all_p_reacts_reacts:
        p_reacts = list(map(str, p_reacts))
        r = maxfrag_k(t_react, p_reacts)
        print(r)

    print()

    t_react = "11"
    for p_reacts in all_p_reacts_reacts:
        p_reacts = list(map(lambda i: f"{i+1}.{i}{i}", p_reacts))
        r = maxfrag_k(t_react, p_reacts)
        print(r)

    print()

    t_react = "11"
    for p_reacts in all_p_reacts_reacts:
        p_reacts = list(map(lambda i: f"{i}.{i}{i+1}", p_reacts))
        r = maxfrag_k(t_react, p_reacts)
        print(r)
