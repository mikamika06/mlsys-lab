import math
from triton_grid.derived import compute_grid


def analyze_configs(configs, shape):
    results = []
    for cfg in configs:
        grid = compute_grid(cfg, shape)
        total_blocks = math.prod(grid)
        warps = cfg.get("num_warps", 4)
        stages = cfg.get("num_stages", 2)
        score = total_blocks * warps * stages
        results.append({
            "config": cfg,
            "grid": grid,
            "total_blocks": total_blocks,
            "score": score
        })
    return results
