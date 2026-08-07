import ref

def check(workdir):
    from servemetrics.engine import simulate_generation
    out = {"simulation_match": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        for guided in [False, True]:
            want = ref.simulate_generation(cfg, guided=guided, seed=42)
            got = simulate_generation(cfg, guided=guided, seed=42)
            if got is not None and len(got) == len(want):
                diff = sum(abs(a - b) for a, b in zip(got, want))
                if diff < 1e-5:
                    ok += 1
                    break
        else:
            if "_note" not in out:
                out["_note"] = f"config {i} simulation mismatch"
    out["simulation_match"] = float(ok)
    return out
