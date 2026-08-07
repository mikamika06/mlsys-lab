import tracemalloc
import numpy as np


def _oracle(Q, K, V, B):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    logits = Q @ K.T / np.sqrt(Q.shape[1]) + B
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)

    Q = rng.normal(size=(19, 8)).astype(np.float64)
    K = rng.normal(size=(19, 8)).astype(np.float64)
    V = rng.normal(size=(19, 6)).astype(np.float64)
    B = rng.normal(size=(19, 19)).astype(np.float64) * 0.2

    try:
        got = sol.mem_efficient_attention(Q.tolist(), K.tolist(), V.tolist(), B.tolist(), 5)
        err = float(np.max(np.abs(np.asarray(got) - _oracle(Q, K, V, B))))
    except Exception:
        return {"max_abs_err": 1e9, "no_n2_materialization": 0.0}

    memory_ok = 1.0
    n = 1536
    Qm = rng.normal(size=(n, 16)).astype(np.float64)
    Km = rng.normal(size=(n, 16)).astype(np.float64)
    Vm = rng.normal(size=(n, 8)).astype(np.float64)
    Bm = rng.normal(size=(n, n)).astype(np.float64) * 0.01

    Qm_list = Qm.tolist()
    Km_list = Km.tolist()
    Vm_list = Vm.tolist()
    Bm_list = Bm.tolist()

    tracemalloc.start()
    try:
        sol.mem_efficient_attention(Qm_list, Km_list, Vm_list, Bm_list, 64)
        _, peak = tracemalloc.get_traced_memory()
    except Exception:
        memory_ok = 0.0
        peak = 0
    finally:
        tracemalloc.stop()

    extra_bytes = max(0, peak - Bm.nbytes)
    dense_temp = n * n * 8
    if extra_bytes > int(dense_temp * 1.25):
        memory_ok = 0.0

    return {
        "max_abs_err": err,
        "no_n2_materialization": memory_ok,
    }
