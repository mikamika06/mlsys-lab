from tritongrid.derivation import compute_grid


def analyze_sweep(problem_shape, configs):
    results = []
    for cfg in configs:
        g = compute_grid(problem_shape, cfg)
        total_blocks = g[0] * g[1]
        results.append({
            "config_id": cfg.get("id"),
            "grid": g,
            "total_blocks": total_blocks
        })
    return results
