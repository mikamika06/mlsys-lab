def grade(sol, fx):
    import numpy as np
    # Test cases: (input_dim, output_dim, hidden_size)
    cases = [
        (768, 768, 2048),
        (512, 256, 1024),
        (128, 64, 512),
    ]
    ratios = []
    for input_dim, output_dim, hidden in cases:
        try:
            vanilla, gated = sol.param_counts(input_dim, output_dim, hidden)
        except Exception:
            return {"param_ratio": 0.0}
        if vanilla <= 0 or gated <= 0:
            return {"param_ratio": 0.0}
        ratios.append(gated / vanilla)
    avg_ratio = float(np.mean(ratios))
    return {"param_ratio": avg_ratio}
