import numpy as np
from attnsink.sink_softmax import attention_sink_softmax


def compute_rel_err(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-12))


def compute_drift(Q: np.ndarray, K: np.ndarray, V: np.ndarray, sink_size: int, window_size: int):
    L = Q.shape[0]
    full_out, _ = attention_sink_softmax(Q, K, V, sink_size=L, window_size=L)
    win_out, _ = attention_sink_softmax(Q, K, V, sink_size=0, window_size=window_size)
    sink_out, _ = attention_sink_softmax(Q, K, V, sink_size=sink_size, window_size=window_size)

    win_rel_err = compute_rel_err(win_out, full_out)
    sink_rel_err = compute_rel_err(sink_out, full_out)

    sink_diff = np.linalg.norm(sink_out - full_out, axis=-1)
    win_diff = np.linalg.norm(win_out - full_out, axis=-1)
    full_norm = np.linalg.norm(full_out, axis=-1) + 1e-12

    return {
        "full_out": full_out,
        "win_out": win_out,
        "sink_out": sink_out,
        "win_rel_err": win_rel_err,
        "sink_rel_err": sink_rel_err,
        "drift_by_pos": sink_diff / full_norm,
        "win_drift_by_pos": win_diff / full_norm,
    }
