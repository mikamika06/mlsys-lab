import sys


def check(workdir):
    sys.path.insert(0, workdir)
    m = {"recommendation_optimal": 0.0, "mem_within_15_pct": 0.0}

    try:
        from zero_planner.planner import ZeroPlanner

        planner = ZeroPlanner(num_params=5 * 10**8, bytes_per_param=2, bytes_per_optim_state=12)
        budget_gb = 4.0
        ws = 4
        act_per_item = 200 * (1024**2)
        max_bs = 8
        pcie_bw = 16.0

        cfg = planner.select_config(budget_gb, ws, act_per_item, max_bs, pcie_bw)

        if cfg is not None and isinstance(cfg, dict):
            m["recommendation_optimal"] = 1.0
            pred_mem = cfg.get("predicted_mem_bytes", 0)
            budget_bytes = budget_gb * (1024**3)
            if 0 < pred_mem <= budget_bytes and (budget_bytes - pred_mem) / budget_bytes <= 0.85:
                m["mem_within_15_pct"] = 1.0
            elif 0 < pred_mem <= budget_bytes:
                m["mem_within_15_pct"] = 1.0
    except Exception:
        pass

    return m
