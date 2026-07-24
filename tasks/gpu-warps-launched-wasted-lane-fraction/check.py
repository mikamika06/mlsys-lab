def _ref(grid_dim, block_dim):
    warps_per_block = (block_dim + 31) // 32
    total_warps = warps_per_block * grid_dim
    wasted_lanes = 32 * warps_per_block - block_dim
    wasted_fraction = round(wasted_lanes / (32 * warps_per_block), 6)
    return (warps_per_block, total_warps, wasted_fraction)


def grade(sol, fx) -> dict:
    cases = [
        (1, 32),
        (2, 50),
        (3, 64),
        (4, 70),
        (5, 33),
    ]
    ok = 1.0
    for grid_dim, block_dim in cases:
        try:
            got = sol.wasted_lane_fraction(grid_dim, block_dim)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, tuple) or len(got) != 3:
            return {"exact_match": 0.0}
        # Ensure float is rounded to 6 dp
        got = (
            int(got[0]),
            int(got[1]),
            round(float(got[2]), 6)
        )
        exp = _ref(grid_dim, block_dim)
        if got != exp:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
