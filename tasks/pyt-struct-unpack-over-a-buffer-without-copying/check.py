import struct

import numpy as np

FAIL = {"exact_match": 0.0, "zero_copy_fraction": 0.0}


def _make_buf(n, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(-10_000, 10_000, size=n).astype(np.int32)
    xs = rng.standard_normal(n).astype(np.float32)
    ys = rng.standard_normal(n).astype(np.float32)
    buf = bytearray()
    for i in range(n):
        buf += struct.pack('<iff', int(ids[i]), float(xs[i]), float(ys[i]))
    return buf, ids, xs, ys


def grade(sol, fx) -> dict:
    n = 40
    buf, ids, xs, ys = _make_buf(n, seed=0)

    try:
        views = sol.parse_records_view(buf, n)
    except Exception:
        return dict(FAIL)

    if not isinstance(views, list) or len(views) != n:
        return dict(FAIL)

    n_ok = 0
    for i, triple in enumerate(views):
        try:
            id_view, x_view, y_view = triple
            if not (isinstance(id_view, memoryview)
                    and isinstance(x_view, memoryview)
                    and isinstance(y_view, memoryview)):
                continue
            if int(id_view[0]) != int(ids[i]):
                continue
            if float(x_view[0]) != float(xs[i]):
                continue
            if float(y_view[0]) != float(ys[i]):
                continue
            n_ok += 1
        except Exception:
            continue

    exact_match = 1.0 if n_ok == n else float(n_ok) / n

    # -- zero-copy check: mutate the ORIGINAL buffer in place after the fact,
    # and confirm the already-returned views observe the new bytes without
    # any further call into the solution.
    mutate_at = [0, n // 2, n - 1]
    new_vals = [(111, 3.5, -3.5), (222, -7.25, 1.25), (333, 0.0, 42.0)]

    n_view_ok = 0
    for idx, (new_id, new_x, new_y) in zip(mutate_at, new_vals):
        off = idx * 12
        buf[off:off + 4] = struct.pack('<i', new_id)
        buf[off + 4:off + 8] = struct.pack('<f', new_x)
        buf[off + 8:off + 12] = struct.pack('<f', new_y)
        try:
            id_view, x_view, y_view = views[idx]
            ok = (int(id_view[0]) == new_id
                  and float(x_view[0]) == np.float32(new_x)
                  and float(y_view[0]) == np.float32(new_y))
        except Exception:
            ok = False
        if ok:
            n_view_ok += 1

    zero_copy_fraction = float(n_view_ok) / len(mutate_at)

    return {"exact_match": exact_match, "zero_copy_fraction": zero_copy_fraction}
