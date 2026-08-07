def compare_framework_timings(mlx_times: list, torch_times: list) -> dict:
    if not mlx_times or not torch_times:
        return {"mean_mlx": 0.0, "mean_torch": 0.0, "speedup": 1.0}
    mean_mlx = sum(mlx_times) / len(mlx_times)
    mean_torch = sum(torch_times) / len(torch_times)
    speedup = mean_torch / mean_mlx if mean_mlx > 0 else 1.0
    return {
        "mean_mlx": float(mean_mlx),
        "mean_torch": float(mean_torch),
        "speedup": float(speedup)
    }
