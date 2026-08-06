def run_server_simulation(run_cfg):
    acc = run_cfg["accepted"]
    tot = run_cfg["total"]
    return {
        "accepted_tokens": acc,
        "total_tokens": tot,
        "acceptance_rate": acc / tot
    }
