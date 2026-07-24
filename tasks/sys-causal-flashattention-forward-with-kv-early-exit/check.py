import sys
import numpy as np


def _dense_causal_reference(Q, K, V):
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(float(d))
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    scores = scores.astype(np.float64)
    scores[mask] = -np.inf
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=1, keepdims=True)
    out = p @ V
    lse = (np.log(np.sum(e, axis=1)) + m[:, 0])
    return out, lse


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    Q = rng.normal(size=(7, 4)).astype(np.float64)
    K = rng.normal(size=(7, 4)).astype(np.float64)
    V = rng.normal(size=(7, 3)).astype(np.float64)

    ref_out, ref_lse = _dense_causal_reference(Q, K, V)

    calls = []

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "_score_kv_tile":
            calls.append(dict(frame.f_locals))
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        got_out, got_lse = sol.causal_flash_attention_forward(
            Q, K, V, tile_size=2
        )
    except Exception:
        got_out = np.full_like(ref_out, np.nan)
        got_lse = np.full_like(ref_lse, np.nan)
    finally:
        sys.settrace(old)

    err = max(
        float(np.max(np.abs(np.asarray(got_out) - ref_out))),
        float(np.max(np.abs(np.asarray(got_lse) - ref_lse))),
    )

    skipped = 0.0
    for loc in calls:
        q_start = int(loc.get("q_start", -1))
        k_start = int(loc.get("k_start", -1))
        tile = int(loc.get("tile_size", 1))
        if k_start >= q_start + tile:
            skipped = 0.0
            break
        skipped = 1.0

    return {
        "max_abs_err": err,
        "skipped_kv_tiles": skipped,
    }
