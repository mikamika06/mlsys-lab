import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from prefixcache.benchmark import benchmark_cache_implementations

    out = {"table_matches": 0.0, "savings_correct": 0.0}

    configs_ok = 0
    savings_ok = 0

    for cfg in ref.BENCHMARK_CONFIGS:
        want = ref.benchmark_cache_implementations(**cfg)
        try:
            got = benchmark_cache_implementations(**cfg)
        except Exception as e:
            out["_note"] = (
                f"benchmark_cache_implementations raised exception: {type(e).__name__}: {str(e)}"
            )
            return out

        if not isinstance(got, dict):
            out["_note"] = "benchmark_cache_implementations did not return a dict"
            return out

        required_types = {"dynamic", "static", "offloaded", "quantized"}
        if set(got.keys()) != required_types:
            out["_note"] = f"keys {set(got.keys())} != expected {required_types}"
            return out

        match_cfg = True
        match_savings = True

        for k in required_types:
            want_item = want[k]
            got_item = got[k]

            for field in ["allocated_bytes", "peak_bytes", "supports_dynamic_growth"]:
                if got_item.get(field) != want_item.get(field):
                    match_cfg = False
                    out["_note"] = (
                        f"impl {k}, field {field}: got {got_item.get(field)}, want {want_item.get(field)}"
                    )
                    break

            want_sav = want_item.get("memory_savings_vs_static", 0.0)
            got_sav = got_item.get("memory_savings_vs_static", 0.0)
            if abs(got_sav - want_sav) > 1e-4:
                match_savings = False
                out["_note"] = (
                    f"impl {k}, memory_savings_vs_static: got {got_sav}, want {want_sav}"
                )

        if match_cfg:
            configs_ok += 1
        if match_savings:
            savings_ok += 1

    if configs_ok == len(ref.BENCHMARK_CONFIGS):
        out["table_matches"] = 1.0
    if savings_ok == len(ref.BENCHMARK_CONFIGS):
        out["savings_correct"] = 1.0

    return out
