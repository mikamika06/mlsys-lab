import numpy as np
import ref


def check(workdir):
    from scoremod import alibi_slopes

    out = {"rel_err": float("inf"), "invariant_ok": 0.0}
    worst = 0.0
    invariant_ok = True
    for n in ref.N_HEADS_LIST:
        want = np.asarray(ref.alibi_slopes(n), dtype=np.float64)
        try:
            got = np.asarray(alibi_slopes(n), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            out["_note"] = f"n_heads={n} raised {type(e).__name__}: {str(e)[:120]}"
            return out
        if got.shape != want.shape:
            out["_note"] = f"n_heads={n}: shape {got.shape} != {want.shape}"
            return out

        denom = np.maximum(np.abs(want), 1e-12)
        err = float(np.max(np.abs(got - want) / denom))
        worst = max(worst, err)

        bounded = bool(np.all(got > 0.0) and np.all(got <= 1.0 + 1e-9))
        distinct = len({round(float(v), 9) for v in got}) == n
        if not (bounded and distinct):
            invariant_ok = False
            if "_note" not in out:
                out["_note"] = f"n_heads={n}: slopes violate bounds/distinctness: {got}"

    out["rel_err"] = worst
    out["invariant_ok"] = 1.0 if invariant_ok else 0.0
    return out
