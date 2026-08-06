def _ref(shape):
    B, H, S, d = shape
    # strides for BHSD layout (C‑order)
    strides_bh = [H * S * d, S * d, d, 1]
    # strides for BSHD layout (C‑order)
    strides_sh = [S * H * d, H * d, d, 1]

    def cache_lines(traversal_shape, strides):
        # number of distinct cache lines accessed during a full nested traversal
        line_size_elems = 8  # 64 bytes / 8 bytes per float64
        seen = set()
        for b in range(traversal_shape[0]):
            for i1 in range(traversal_shape[1]):
                for i2 in range(traversal_shape[2]):
                    for i3 in range(traversal_shape[3]):
                        offset = (
                            b * strides[0]
                            + i1 * strides[1]
                            + i2 * strides[2]
                            + i3 * strides[3]
                        )
                        seen.add(offset // line_size_elems)
        return len(seen)

    cost_bh = cache_lines((B, H, S, d), strides_bh)
    cost_sh = cache_lines((B, S, H, d), strides_sh)
    return cost_bh, cost_sh

def grade(sol, fx) -> dict:
    # Test a handful of shapes
    tests = [
        (2, 3, 4, 5),
        (1, 8, 16, 32),
        (4, 2, 6, 7),
        (10, 10, 10, 10),
        (3, 5, 9, 11)
    ]
    ok = 1.0
    for shape in tests:
        try:
            got = sol.layout_cost(shape)
            ref = _ref(shape)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, tuple) or len(got) != 2:
            return {"exact_match": 0.0}
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
