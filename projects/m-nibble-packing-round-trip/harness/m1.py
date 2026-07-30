import numpy as np
import ref


def check(workdir):
    from nibblepack import pack_nibbles, unpack_nibbles

    out = {"byte_exact_fraction": 0.0, "cases": float(len(ref.CASES))}
    ok = 0
    for i, codes in enumerate(ref.CASES):
        n = len(codes)
        want_packed = ref.pack_nibbles(codes)
        try:
            got_packed = np.asarray(pack_nibbles(codes), dtype=np.uint8).reshape(-1)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} (n={n}): pack_nibbles raised {type(e).__name__}: {str(e)[:120]}"
            continue
        pack_ok = got_packed.shape == want_packed.shape and np.array_equal(got_packed, want_packed)

        try:
            got_back = np.asarray(unpack_nibbles(want_packed, n), dtype=np.uint8).reshape(-1)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} (n={n}): unpack_nibbles raised {type(e).__name__}: {str(e)[:120]}"
            continue
        want_codes = np.array(codes, dtype=np.uint8)
        unpack_ok = got_back.shape == want_codes.shape and np.array_equal(got_back, want_codes)

        if pack_ok and unpack_ok:
            ok += 1
        elif "_note" not in out:
            out["_note"] = (f"case {i} (n={n}): pack_match={pack_ok} unpack_match={unpack_ok} "
                             f"got_packed={got_packed.tolist()[:6]} want_packed={want_packed.tolist()[:6]}")

    out["byte_exact_fraction"] = ok / len(ref.CASES) if ref.CASES else 1.0
    return out
