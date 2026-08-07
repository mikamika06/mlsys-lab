import ref
import numpy as np

def check(workdir):
    from prune.eval import evaluate_mse

    m = {"eval_ok": 0.0}
    w, x = ref.get_fixture()
    w_p = w.copy()
    w_p[:, :32] = 0.0
    bias = np.ones(w.shape[0])

    val = evaluate_mse(w, w_p, bias, x)
    y_target = w @ x
    y_pred = w_p @ x + bias[:, None]
    expected_val = float(np.mean((y_target - y_pred)**2))

    if isinstance(val, float) and np.isclose(val, expected_val, rtol=1e-4):
        m["eval_ok"] = 1.0

    return m
