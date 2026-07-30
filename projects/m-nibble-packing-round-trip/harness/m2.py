import numpy as np
import ref


def check(workdir):
    from nibblepack import dequantize_block

    out = {"values_match": 0.0, "zero_absmax_is_zero": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    zero_ok = 0
    zero_total = 0
    for i, (codes, absmax) in enumerate(ref.CONFIGS):
        n = len(codes)
        is_zero = (absmax == 0.0)
        if is_zero:
            zero_total += 1
        packed = ref.pack_nibbles(codes)
        want = ref.dequantize_block(packed, n, absmax)
        try:
            got = np.asarray(dequantize_block(packed, n, absmax), dtype=np.float64).reshape(-1)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"config {i} (n={n}, absmax={absmax}): raised {type(e).__name__}: {str(e)[:120]}"
            continue
        match = got.shape == want.shape and np.allclose(got, want, rtol=1e-9, atol=1e-9)
        if match:
            ok += 1
            if is_zero:
                zero_ok += 1
        elif "_note" not in out:
            out["_note"] = (f"config {i} (n={n}, absmax={absmax}): "
                             f"got={got.tolist()[:4]} want={want.tolist()[:4]}")

    out["values_match"] = ok / len(ref.CONFIGS) if ref.CONFIGS else 1.0
    out["zero_absmax_is_zero"] = (zero_ok / zero_total) if zero_total else 1.0
    return out
