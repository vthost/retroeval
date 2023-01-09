

def rec_k(t_reacts, p_reacts, ks=[1, 5, 10]):
    n = len(t_reacts)
    ks = sorted(ks)
    top_ks = [0]*len(ks)
    for t_react in t_reacts:
        for i, k in enumerate(ks):
            if t_react in p_reacts[ks[i-1] if i else 0:k]:
                top_ks[i] += 1 / n
                if k < ks[-1]:
                    for j, k2 in enumerate(ks[i + 1:]):
                        top_ks[i + 1 + j] += 1 / n
                    break
    return top_ks
