import numpy as np


def _oracle(master_weight, grad_fp16, lr):
    master32 = np.asarray(master_weight, dtype=np.float32)
    grad32 = np.asarray(grad_fp16, dtype=np.float32)
    updated_master = master32 - np.float32(lr) * grad32
    updated_model = updated_master.astype(np.float16)
    return updated_master, updated_model


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, -2.0, 3.5], dtype=np.float32),
            np.array([0.5, 0.25, -1.25], dtype=np.float16),
            0.1,
        ),
        (
            np.array([10000.0, -5000.5, 0.125], dtype=np.float32),
            np.array([3.25, -4.5, 0.75], dtype=np.float16),
            0.003,
        ),
        (
            np.array([1e-3, -2e-3, 4e-3, 8e-3], dtype=np.float32),
            np.array([0.001, -0.002, 0.003, -0.004], dtype=np.float16),
            0.25,
        ),
    ]

    worst = 0.0
    for w, g, lr in cases:
        try:
            got_master, got_model = sol.mixed_precision_step(w, g, lr)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref_master, ref_model = _oracle(w, g, lr)
        err = max(
            float(np.max(np.abs(np.asarray(got_master, dtype=np.float32) - ref_master))),
            float(np.max(np.abs(np.asarray(got_model, dtype=np.float16) - ref_model))),
        )
        worst = max(worst, err)

    return {"max_abs_err": worst}
