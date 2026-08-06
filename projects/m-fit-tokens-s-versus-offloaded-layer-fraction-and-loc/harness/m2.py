import sys

import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from offload.memory import fit_layers_in_budget

    out = {"fit_exact_match": 0.0, "respects_memory_cap": 0.0}

    budgets = [
        1000 * 1024 * 1024,
        2000 * 1024 * 1024,
        4000 * 1024 * 1024,
        10000 * 1024 * 1024,
    ]

    exact_matches = 0
    total_tests = 0
    memory_safe = True

    for cfg in ref.CONFIGS:
        for b in budgets:
            total_tests += 1
            want = ref.fit_layers_in_budget(cfg, b)
            got = fit_layers_in_budget(cfg, b)

            if got == want:
                exact_matches += 1
            elif "_note" not in out:
                out["_note"] = f"Config total_layers={cfg['total_layers']}, budget={b}: want {want}, got {got}"

            req_mem = cfg["base_overhead_bytes"] + got * (
                cfg["bytes_per_layer_weight"] + cfg["bytes_per_layer_kv"]
            )
            if req_mem > b and got > 0:
                memory_safe = False

    if exact_matches == total_tests:
        out["fit_exact_match"] = 1.0
    if memory_safe:
        out["respects_memory_cap"] = 1.0

    return out
