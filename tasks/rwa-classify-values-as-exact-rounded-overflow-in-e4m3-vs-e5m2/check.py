import numpy as np


def _decode_e4m3fn_one(code: int) -> float:
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0x0F
    m = code & 0x07
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return float("nan")
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def _grid_e4m3fn() -> np.ndarray:
    vals = set()
    for code in range(0, 128):  # nonneg codes only
        if code == 0x7F:
            continue  # nan
        vals.add(_decode_e4m3fn_one(code))
    return np.array(sorted(vals), dtype=np.float64)


def _decode_e5m2_vec(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    e = ((codes >> 2) & 0x1F).astype(np.int64)
    m = (codes & 0x3).astype(np.int64)
    normal_val = np.exp2((e - 15).astype(np.float64)) * (1.0 + m / 4.0)
    sub_val = np.exp2(-14.0) * (m / 4.0)
    return np.where(e == 0, sub_val, normal_val)


def _grid_e5m2() -> np.ndarray:
    codes = np.arange(0, 124, dtype=np.uint8)  # exclude e=31 block (inf/nan)
    vals = _decode_e5m2_vec(codes)
    return np.sort(np.unique(vals))


_GRID_E4M3 = _grid_e4m3fn()
_MAX_E4M3 = float(_GRID_E4M3[-1])
_GRID_E5M2 = _grid_e5m2()
_MAX_E5M2 = float(_GRID_E5M2[-1])


def _ref_labels(values: np.ndarray, grid: np.ndarray, max_finite: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    av = np.abs(values)
    labels = np.empty(values.shape, dtype="<U8")
    overflow = av > max_finite
    labels[overflow] = "overflow"
    inrange = ~overflow
    exact_mask = np.isin(av, grid)
    labels[inrange & exact_mask] = "exact"
    labels[inrange & ~exact_mask] = "rounded"
    return labels


def _build_values() -> np.ndarray:
    rng = np.random.default_rng(0)

    exact_e4m3 = _GRID_E4M3[::5]
    exact_e5m2 = _GRID_E5M2[::7]

    eps = 1e-3
    rounded_e4m3 = _GRID_E4M3[1:-1:6] + eps
    rounded_e5m2 = _GRID_E5M2[1:-1:9] + eps

    # overflow relative to e4m3 but in-range for e5m2
    cross = np.array([500.0, 1000.0, 4000.0, 40000.0, -600.0])
    # overflow relative to both
    both_over = np.array([1e6, -1e6, 1e9])

    edges = np.array([0.0, -0.0, _MAX_E4M3, -_MAX_E4M3, _MAX_E5M2, -_MAX_E5M2,
                       _MAX_E4M3 + 1e-6, _MAX_E5M2 + 1e-6])

    rand = rng.uniform(-70000.0, 70000.0, size=500)

    x = np.concatenate([
        exact_e4m3, exact_e5m2, rounded_e4m3, rounded_e5m2,
        cross, both_over, edges, rand,
    ]).astype(np.float64)
    return x


def grade(sol, fx) -> dict:
    values = _build_values()

    exact = 1.0
    for grid, max_finite in [(_GRID_E4M3, _MAX_E4M3), (_GRID_E5M2, _MAX_E5M2)]:
        ref = _ref_labels(values, grid, max_finite)
        try:
            got = sol.classify_fp8(values.copy(), grid.copy(), max_finite)
        except Exception:
            return {"exact_match": 0.0}

        got = np.asarray(got)
        if got.shape != ref.shape:
            return {"exact_match": 0.0}
        try:
            got_str = got.astype("<U8")
        except Exception:
            return {"exact_match": 0.0}

        if not np.array_equal(got_str, ref):
            exact = 0.0
            break

    return {"exact_match": exact}
