import ref


def check(workdir):
    from moeroof import pack_experts
    cases = ref.get_packing_cases()
    ok = 0
    out = {"packing_matched": 0.0}
    for i, c in enumerate(cases):
        want_gpus = ref.pack_experts(c["loads"], c["num_gpus"])
        want_max_load = max(sum(c["loads"][idx] for idx in g) for g in want_gpus)
        got_gpus = pack_experts(c["loads"], c["num_gpus"])
        if not isinstance(got_gpus, list) or len(got_gpus) != c["num_gpus"]:
            continue
        got_flat = [idx for g in got_gpus for idx in g]
        if sorted(got_flat) != sorted(range(len(c["loads"]))):
            continue
        got_max_load = max(sum(c["loads"][idx] for idx in g) for g in got_gpus)
        if got_max_load == want_max_load:
            ok += 1
    if ok == len(cases):
        out["packing_matched"] = 1.0
    return out
