import math
import numpy as np


def grade(sol, fx):
    np.random.seed(1337)
    d = 8
    H = 4
    head = 1
    n = 2

    Wq = np.random.randn(d, d).tolist()
    Wk = np.random.randn(d, d).tolist()
    Wv = np.random.randn(d, d).tolist()
    Wo = np.random.randn(d, d).tolist()
    x = np.random.randn(n, d).tolist()

    try:
        res = sol.remove_attention_head(Wq, Wk, Wv, Wo, x, head, H)
    except Exception as e:
        return {"max_abs_err": float("inf"), "error": str(e)}

    if not isinstance(res, tuple) or len(res) != 5:
        return {"max_abs_err": float("inf"), "error": "Invalid return type or length"}

    Wq_p_sol, Wk_p_sol, Wv_p_sol, Wo_p_sol, y_sol = res

    # Oracle computation using numpy
    Wq_np = np.array(Wq)
    Wk_np = np.array(Wk)
    Wv_np = np.array(Wv)
    Wo_np = np.array(Wo)
    x_np = np.array(x)

    head_dim = d // H
    start = head * head_dim
    end = (head + 1) * head_dim

    oracle_Wq_p = np.concatenate([Wq_np[:, :start], Wq_np[:, end:]], axis=1)
    oracle_Wk_p = np.concatenate([Wk_np[:, :start], Wk_np[:, end:]], axis=1)
    oracle_Wv_p = np.concatenate([Wv_np[:, :start], Wv_np[:, end:]], axis=1)
    oracle_Wo_p = np.concatenate([Wo_np[:start], Wo_np[end:]], axis=0)

    q = x_np @ oracle_Wq_p
    k = x_np @ oracle_Wk_p
    v = x_np @ oracle_Wv_p

    outputs_oracle = []
    scale = math.sqrt(head_dim)
    rem_idx = 0
    for i in range(H):
        if i == head:
            continue
        a = rem_idx * head_dim
        b = (rem_idx + 1) * head_dim
        rem_idx += 1

        qi = q[:, a:b]
        ki = k[:, a:b]
        vi = v[:, a:b]

        scores = (qi @ ki.T) / scale
        max_vals = np.max(scores, axis=1, keepdims=True)
        exps = np.exp(scores - max_vals)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        head_out = probs @ vi
        outputs_oracle.append(head_out)

    concat_oracle = np.concatenate(outputs_oracle, axis=1)
    y_oracle = concat_oracle @ oracle_Wo_p

    y_sol_np = np.array(y_sol)

    if y_sol_np.shape != y_oracle.shape:
        return {
            "max_abs_err": float("inf"),
            "error": f"Shape mismatch: {y_sol_np.shape} vs {y_oracle.shape}",
        }

    max_err = float(np.max(np.abs(y_sol_np - y_oracle)))

    matrices_err = max(
        float(np.max(np.abs(np.array(Wq_p_sol) - oracle_Wq_p))),
        float(np.max(np.abs(np.array(Wk_p_sol) - oracle_Wk_p))),
        float(np.max(np.abs(np.array(Wv_p_sol) - oracle_Wv_p))),
        float(np.max(np.abs(np.array(Wo_p_sol) - oracle_Wo_p))),
    )

    total_err = max(max_err, matrices_err)
    return {"max_abs_err": total_err}
