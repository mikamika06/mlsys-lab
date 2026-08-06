def compute_eagle_metrics(run_cfg, sim_out):
    ar = sim_out["acceptance_rate"]
    lat_ratio = run_cfg["eagle_tpot"] / run_cfg["baseline_tpot"]
    tpot_gain = run_cfg["baseline_tpot"] / run_cfg["eagle_tpot"]
    return {
        "acceptance_rate": ar,
        "latency_ratio": lat_ratio,
        "tpot_gain": tpot_gain
    }
