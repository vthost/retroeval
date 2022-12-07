
def _eval_mss_level(t_reacts, p_reacts, topk=10):
    pos_idx, neg_idx = [], []
    for i, t_react in enumerate(t_reacts):
        if t_react in p_reacts[i][:topk]:
            pos_idx += [i]
        else:
            neg_idx += [i]
    return pos_idx, neg_idx


# - this method assumes that the saved predictions are in the same order as the dataframe df
# - also note, our data contains reactions only once per route
# therefore the code is in line with the paper's def which considers sets
# - option to return a second, more strict score, which only counts reactions in routes solved entirely
def mss(t_df, p_reacts, topk=10, strict=False, ks=[], **kwargs):
    if ks:
        topk = max(ks)  # TODO this is a slight workaround to for the deficiency that it doesn't cover all ks

    # walk through all levels from target at root to leaves
    already_failed = []  # record a route that fails at some point, is then ignored at later levels
    all_pos = 0
    for level in range(max(t_df['level']) + 1):
        df_level = t_df[t_df['level'] == level]

        idx_failed = df_level['route_id'].isin(already_failed)
        df_level = df_level[~idx_failed]  # the ones to be checked for current level

        t_reacts_level = df_level['reactants'].values.tolist()
        p_reacts_level = [p for i, p in enumerate(p_reacts) if i in df_level.index]
        pos_idx, neg_idx = _eval_mss_level(t_reacts_level, p_reacts_level, topk=topk)

        all_pos += len(pos_idx)
        already_failed += df_level.iloc[neg_idx]['route_id'].drop_duplicates().values.tolist()

    if strict:
        idx = t_df['route_id'].isin(already_failed)
        df_pos_strict = t_df[~idx]
        return len(df_pos_strict.index) / len(t_df.index)

    return [all_pos / len(t_df.index)]