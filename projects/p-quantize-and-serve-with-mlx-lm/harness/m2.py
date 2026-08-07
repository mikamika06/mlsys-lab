import numpy as np

def check(workdir):
    import ref
    from mlx_serve.quality import measure_quality
    orig = np.array([1.0, 2.0, 3.0])
    quant = np.array([1.1, 1.9, 3.1])
    diff = measure_quality(orig, quant)
    ref_diff = ref.compute_perplexity_diff(orig, quant)
    ok = 1.0 if abs(diff - ref_diff) < 1e-5 else 0.0
    return {"perplexity_diff_ok": ok}
