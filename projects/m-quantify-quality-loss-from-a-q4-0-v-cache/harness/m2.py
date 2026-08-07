import numpy as np
import ref


def check(workdir):
    from vcache.quant_eval import evaluate_v_cache_loss

    out = {"q4_err_matched": 0.0, "rel_err": 1.0}

    np.random.seed(123)
    test_matrices = [
        np.random.randn(2, 4, 64, 32).astype(np.float32),
        np.random.randn(4, 8, 128, 64).astype(np.float32),
        np.random.uniform(-5.0, 5.0, size=(1, 2, 256, 128)).astype(np.float32),
        np.random.randn(8, 4, 32, 32).astype(np.float32)
    ]

    matched = 0
    max_rel_err = 0.0
    for i, mat in enumerate(test_matrices):
        want = ref.evaluate_v_cache_loss(mat)
        try:
            got = evaluate_v_cache_loss(mat)
        except Exception as e:
            out["_note"] = f"Matrix {i} raised {type(e).__name__}: {e}"
            return out

        err = abs(want - got) / (abs(want) + 1e-8)
        if err > max_rel_err:
            max_rel_err = err

        if err <= 0.05:
            matched += 1

    out["q4_err_matched"] = float(matched)
    out["rel_err"] = float(max_rel_err)
    return out
