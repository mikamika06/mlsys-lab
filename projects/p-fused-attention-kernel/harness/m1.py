import numpy as np
import ref


def check(workdir):
    from fused_attn.kernel import online_softmax_step
    m = {"online_softmax_ok": 0.0}
    chunk = np.array([[1.0, 2.0, 3.0]])
    m_prev = np.array([[-1.0]])
    d_prev = np.array([[1.0]])
    try:
        m_new, d_new, exp_chunk = online_softmax_step(chunk, m_prev, d_prev)
        expected_m = np.array([[3.0]])
        if np.allclose(m_new, expected_m):
            m["online_softmax_ok"] = 1.0
    except Exception:
        pass
    return m
