import ref

def check(workdir):
    from moe_sim.metrics import latency, sweep_configs

    out = {"latency_match": 0.0, "throughput_ratio": 0.0}

    try:
        match = True
        for ngl in [0, 10, 32]:
            for c in [0, 2, 8]:
                if abs(latency(ref.CFG, ngl, c) - ref.latency(ref.CFG, ngl, c)) > 1e-6:
                    match = False
        if match:
            out["latency_match"] = 1.0
    except NotImplementedError:
        pass

    try:
        got = sweep_configs(ref.CFG, 15000)
        want = ref.sweep_configs(ref.CFG, 15000)

        if len(got) == len(want):
            match_all = True
            for g, w in zip(got, want):
                if g["n_cpu_experts"] != w["n_cpu_experts"] or g["ngl"] != w["ngl"]:
                    match_all = False
                if abs(g["vram"] - w["vram"]) > 1e-6:
                    match_all = False
                if abs(g["throughput"] - w["throughput"]) > 1e-6:
                    match_all = False
            if match_all:
                out["throughput_ratio"] = 1.0
    except NotImplementedError:
        pass

    return out
