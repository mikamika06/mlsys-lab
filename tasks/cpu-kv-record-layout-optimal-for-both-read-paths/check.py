from mlsys.sim import cache as cachesim


LINE_BYTES = 64
SETS = 32
WAYS = 4

LAYOUTS = ("THKD", "TKHD", "TDHK", "HTKD", "HKTD")


def _offset(layout, t, h, k, d, T, H, D, E, base):
    if layout == "THKD":
        index = (((t * H + h) * 2 + k) * D + d)
    elif layout == "TKHD":
        index = (((t * 2 + k) * H + h) * D + d)
    elif layout == "TDHK":
        index = (((t * D + d) * H + h) * 2 + k)
    elif layout == "HTKD":
        index = (((h * T + t) * 2 + k) * D + d)
    elif layout == "HKTD":
        index = (((h * 2 + k) * T + t) * D + d)
    else:
        raise ValueError("unknown layout")
    return base + index * E


def _traces_for_layout(layout, T, H, D, E, base):
    newest = T - 1
    write_addrs = []
    for h in range(H):
        for k in range(2):
            for d in range(D):
                write_addrs.append(_offset(layout, newest, h, k, d, T, H, D, E, base))

    read_addrs = []
    for h in range(H):
        for t in range(T):
            for k in range(2):
                for d in range(D):
                    read_addrs.append(_offset(layout, t, h, k, d, T, H, D, E, base))

    return write_addrs, read_addrs


def _miss_count(addrs):
    result = cachesim.simulate(addrs, line_bytes=LINE_BYTES, sets=SETS, ways=WAYS)

    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result
    if isinstance(result, float):
        return int(result)

    if isinstance(result, dict):
        for key in (
            "misses",
            "miss_count",
            "num_misses",
            "load_misses",
            "read_misses",
            "total_misses",
        ):
            if key in result:
                return int(result[key])

    for key in (
        "misses",
        "miss_count",
        "num_misses",
        "load_misses",
        "read_misses",
        "total_misses",
    ):
        if hasattr(result, key):
            return int(getattr(result, key))

    if isinstance(result, (tuple, list)) and result:
        for item in result:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                return int(item)

    raise TypeError("could not extract miss count from cachesim result")


def _traffic_bytes(addrs):
    return _miss_count(list(addrs)) * LINE_BYTES


def _span_lines(addrs):
    if not addrs:
        return 0
    lo = min(addrs) // LINE_BYTES
    hi = max(addrs) // LINE_BYTES
    return hi - lo + 1


def _reference(T, H, D, E, base):
    best = None

    for rank, layout in enumerate(LAYOUTS):
        write_addrs, read_addrs = _traces_for_layout(layout, T, H, D, E, base)
        write_bytes = _traffic_bytes(write_addrs)
        read_bytes = _traffic_bytes(read_addrs)
        span = _span_lines(write_addrs)

        score = (
            write_bytes + read_bytes,
            write_bytes,
            read_bytes,
            span,
            rank,
        )
        record = {
            "layout_id": layout,
            "write_bytes": write_bytes,
            "read_bytes": read_bytes,
            "score": score,
        }
        if best is None or score < best["score"]:
            best = record

    return best


def _as_addr_list(x):
    if isinstance(x, (bytes, bytearray, str)):
        raise TypeError("address trace must be a sequence of integers")
    return [int(v) for v in x]


def _candidate(sol, T, H, D, E, base):
    out = sol.kv_record_layout_trace(T, H, D, E, base)
    if not isinstance(out, dict):
        raise TypeError("return value must be a dict")

    layout_id = str(out["layout_id"])
    write_addrs = _as_addr_list(out["write_addrs"])
    read_addrs = _as_addr_list(out["read_addrs"])

    expected_write_len = H * 2 * D
    expected_read_len = T * H * 2 * D
    if len(write_addrs) != expected_write_len:
        raise ValueError("wrong write trace length")
    if len(read_addrs) != expected_read_len:
        raise ValueError("wrong read trace length")

    if any(a < 0 for a in write_addrs) or any(a < 0 for a in read_addrs):
        raise ValueError("addresses must be nonnegative")

    return {
        "layout_id": layout_id,
        "write_bytes": _traffic_bytes(write_addrs),
        "read_bytes": _traffic_bytes(read_addrs),
    }


def grade(sol, fx) -> dict:
    cases = [
        (32, 8, 16, 2, 0),
        (17, 6, 8, 4, 4096),
        (41, 4, 32, 1, 8192),
        (9, 12, 16, 2, 64),
    ]

    exact = 1.0
    worst_rel = 0.0

    for T, H, D, E, base in cases:
        try:
            ref = _reference(T, H, D, E, base)
            got = _candidate(sol, T, H, D, E, base)

            diff = abs(got["write_bytes"] - ref["write_bytes"])
            diff += abs(got["read_bytes"] - ref["read_bytes"])
            denom = ref["write_bytes"] + ref["read_bytes"] + 1e-12
            rel = diff / denom
            worst_rel = max(worst_rel, float(rel))

            if got["layout_id"] != ref["layout_id"]:
                exact = 0.0
            if got["write_bytes"] != ref["write_bytes"]:
                exact = 0.0
            if got["read_bytes"] != ref["read_bytes"]:
                exact = 0.0
        except Exception:
            exact = 0.0
            worst_rel = float("inf")
            break

    return {"exact_match": exact, "byte_rel_err": worst_rel}
