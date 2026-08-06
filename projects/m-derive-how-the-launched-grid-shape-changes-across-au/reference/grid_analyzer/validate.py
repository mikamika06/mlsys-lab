from grid_analyzer.core import derive_grid

def compare_grids(M, N, configs):
    results = []
    for cfg in configs:
        g = derive_grid(M, N, cfg)
        results.append({
            "config": cfg,
            "grid": g,
            "total_blocks": g[0] * g[1]
        })
    return results
