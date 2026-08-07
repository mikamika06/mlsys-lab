import ref


def check(workdir):
    from oversub.scheduler import sweep_stream_count

    out = {"knee_points_matched": 0.0, "total_scenarios": float(len(ref.SCENARIOS))}
    matched = 0

    for i, sc in enumerate(ref.SCENARIOS):
        bench = ref.make_bench_fn(sc["num_cores"], sc["scaling"], sc["penalty"])
        want_res, want_knee = ref.reference_sweep(bench, sc["max_streams"])

        try:
            got_res, got_knee = sweep_stream_count(bench, sc["max_streams"])
            if got_knee == want_knee and isinstance(got_res, dict):
                res_ok = True
                for k, v in want_res.items():
                    if k not in got_res or abs(got_res[k] - v) > 1e-4:
                        res_ok = False
                        break
                if res_ok:
                    matched += 1
                elif "_note" not in out:
                    out["_note"] = f"scenario {i}: dict values mismatched"
            elif "_note" not in out:
                out["_note"] = f"scenario {i}: got knee {got_knee}, expected {want_knee}"
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"scenario {i} raised {type(e).__name__}: {e}"

    out["knee_points_matched"] = float(matched)
    return out
