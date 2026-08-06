import ref


def check(workdir):
    from tritoncache.manager import JITCacheManager

    out = {"cache_stats_match": 0.0}
    cases = ref.get_test_cases()

    ok = True
    for case in cases:
        mgr = JITCacheManager(case["fn_name"], case["sig"])
        expected_hits = 0
        expected_misses = 0

        seen_keys = set()
        for run in case["runs"]:
            res, compiled = mgr.get_or_compile(run)

            parts = [f"fn:{case['fn_name']}"]
            for param_name, meta in case["sig"].items():
                is_constexpr = meta.get("is_constexpr", False)
                is_ptr = meta.get("is_ptr", False)
                val = run[param_name]
                if is_constexpr:
                    c = ("constexpr", param_name, repr(val))
                elif is_ptr:
                    c = (
                        "tensor",
                        param_name,
                        str(getattr(val, "dtype", "ptr")),
                        tuple(getattr(val, "shape", ())),
                        tuple(getattr(val, "stride", ())),
                    )
                else:
                    c = ("scalar", param_name, type(val).__name__)
                parts.append(f"{param_name}:{c}")
            key = "|".join(parts)

            if key in seen_keys:
                expected_hits += 1
                if compiled:
                    ok = False
                    out["_note"] = "Expected hit but got compile=True"
            else:
                expected_misses += 1
                seen_keys.add(key)
                if not compiled:
                    ok = False
                    out["_note"] = "Expected miss but got compile=False"

        st = mgr.stats()
        if (
            st["hits"] != expected_hits
            or st["misses"] != expected_misses
            or st["cache_size"] != len(seen_keys)
        ):
            ok = False
            out["_note"] = f"Stats mismatch. Got {st}, want hits={expected_hits}, misses={expected_misses}, size={len(seen_keys)}"

    if ok:
        out["cache_stats_match"] = 1.0
    return out
