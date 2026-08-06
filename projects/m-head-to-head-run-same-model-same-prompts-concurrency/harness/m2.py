import ref


def check(workdir):
    from enginebench.metrics import compute_throughput_ratio
    from enginebench.runner import execute_run

    ok = 0
    total = 0
    for conc in [1, 8, 32]:
        prompts = ["prompt one", "prompt two", "prompt three"]
        lat_base = ref.execute_run("engine_a", prompts, conc)
        lat_cand = ref.execute_run("engine_b", prompts, conc)
        num_tokens = 150

        want = ref.compute_throughput_ratio(lat_base, lat_cand, num_tokens, conc)
        try:
            got = compute_throughput_ratio(lat_base, lat_cand, num_tokens, conc)
            if abs(float(got) - float(want)) < 1e-5:
                ok += 1
        except Exception:
            pass
        total += 1

    out = {"throughput_ratio_match": 1.0 if ok == total else 0.0}
    if out["throughput_ratio_match"] == 0.0:
        out["_note"] = f"throughput ratio calculation mismatch ({ok}/{total})"
    return out
