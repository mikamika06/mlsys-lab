import numpy as np

from mlsys import scorers


def _oracle(x):
    x = np.asarray(x, dtype=np.float64)
    blocks = x.reshape(-1, 32)
    B = blocks.shape[0]

    idx = np.argmax(np.abs(blocks), axis=1)
    max_signed = blocks[np.arange(B), idx]
    d = max_signed / -8.0
    d16 = d.astype(np.float16)
    dq = d16.astype(np.float64)

    safe_dq = np.where(dq == 0.0, 1.0, dq)
    q = np.round(blocks / safe_dq[:, None]) + 8.0
    nibbles = np.clip(q, 0, 15).astype(np.uint8)
    nibbles = np.where(dq[:, None] == 0.0, np.uint8(8), nibbles)

    dequant = (nibbles.astype(np.float64) - 8.0) * dq[:, None]

    return d16, nibbles, dequant


def grade(sol, fx) -> dict:
    x = fx["gguf_w"]
    ref_scale, ref_nibbles, ref_dequant = _oracle(x)

    try:
        got = sol.q4_0_block_pack_unpack(x.copy())
        got_scale = np.asarray(got["scale"], dtype=np.float16)
        got_nibbles = np.asarray(got["nibbles"])
        got_dequant = np.asarray(got["dequant"], dtype=np.float64)
    except Exception:
        return {"nibble_exact": 0.0, "rel_err": float("inf")}

    if (
        got_scale.shape != ref_scale.shape
        or got_nibbles.shape != ref_nibbles.shape
        or got_dequant.shape != ref_dequant.shape
    ):
        return {"nibble_exact": 0.0, "rel_err": float("inf")}

    nibble_exact = float(np.mean(got_nibbles.astype(np.int64) == ref_nibbles.astype(np.int64)))
    rel = scorers.rel_err(ref_dequant, got_dequant)

    return {"nibble_exact": nibble_exact, "rel_err": rel}
