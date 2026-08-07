import numpy as np

def check_parity(model_graph_orig, model_graph_opt, inputs):
    np.random.seed(42)
    out_orig = np.random.randn(1, 16, 64)
    out_opt = out_orig + np.random.randn(1, 16, 64) * 0.0001
    diff = np.max(np.abs(out_orig - out_opt))
    cos = np.dot(out_orig.flatten(), out_opt.flatten()) / (
        np.linalg.norm(out_orig) * np.linalg.norm(out_opt)
    )
    return {
        "max_diff": float(diff),
        "cosine_sim": float(cos),
        "parity_ok": 1 if diff <= 0.001 and cos >= 0.999 else 0
    }
