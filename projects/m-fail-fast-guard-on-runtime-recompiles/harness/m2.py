import ref


def check(workdir):
    from recompile.cache import validate_cache, compute_cold_start_cost
    out = {"cache_matched": 0.0}
    ok = 0
    for item in ref.CACHE_TESTS:
        want_valid = set(item["cache_files"]).issubset(set(item["available"]))
        got_valid = validate_cache(item["cache_files"], item["available"])
        want_cost = 1.0 if want_valid else 100.0
        got_cost = compute_cold_start_cost(got_valid)
        if got_valid == want_valid and got_cost == want_cost:
            ok += 1
    out["cache_matched"] = float(ok)
    return out
