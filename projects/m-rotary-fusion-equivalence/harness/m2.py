import ref

def check(workdir):
    from rotary.sweep import optimal_num_splits
    lengths = [128, 256, 1024, 4096]
    ok = 1
    for l in lengths:
        if optimal_num_splits(l) != ref.ref_optimal_num_splits(l):
            ok = 0
            break
    return {"split_match": float(ok)}
