import numpy as np

from mlsys import scorers


def _oracle_dequant(ql, qh, scales, d):
    """NumPy port of GGML's dequantize_row_q6_K for one 256-element super-block."""
    ql = np.asarray(ql, dtype=np.int64)
    qh = np.asarray(qh, dtype=np.int64)
    sc = np.asarray(scales, dtype=np.int64)
    d = float(d)

    y = np.zeros(256, dtype=np.float64)
    l = np.arange(32)
    is_ = l // 16  # 0 for l<16, 1 for l>=16

    for c in range(2):
        qlc = ql[64 * c: 64 * c + 64]
        qhc = qh[32 * c: 32 * c + 32]
        scc = sc[8 * c: 8 * c + 8]

        q1 = (qlc[l] & 0xF | ((qhc[l] >> 0) & 3) << 4) - 32
        q2 = (qlc[l + 32] & 0xF | ((qhc[l] >> 2) & 3) << 4) - 32
        q3 = (qlc[l] >> 4 | ((qhc[l] >> 4) & 3) << 4) - 32
        q4 = (qlc[l + 32] >> 4 | ((qhc[l] >> 6) & 3) << 4) - 32

        yc = np.zeros(128, dtype=np.float64)
        yc[l] = d * scc[is_ + 0] * q1
        yc[l + 32] = d * scc[is_ + 2] * q2
        yc[l + 64] = d * scc[is_ + 4] * q3
        yc[l + 96] = d * scc[is_ + 6] * q4

        y[128 * c: 128 * c + 128] = yc

    return y


def _cases(rng: np.random.Generator):
    cases = []
    for _ in range(2):
        ql = rng.integers(0, 256, size=128, dtype=np.uint8)
        qh = rng.integers(0, 256, size=64, dtype=np.uint8)
        scales = rng.integers(-32, 32, size=16, dtype=np.int8)
        d = float(rng.uniform(0.001, 0.05))
        cases.append((ql, qh, scales, d))
    return cases


def grade(sol, fx) -> dict:
    all_cases = [(
        fx["q6k_ql"], fx["q6k_qh"], fx["q6k_scales"], float(fx["q6k_d"])
    )]
    rng = np.random.default_rng(0)
    all_cases.extend(_cases(rng))

    worst = 0.0
    for ql, qh, scales, d in all_cases:
        expected = _oracle_dequant(ql, qh, scales, d)
        try:
            got = np.asarray(
                sol.dequant_q6_k_superblock(
                    np.array(ql, copy=True), np.array(qh, copy=True),
                    np.array(scales, copy=True), d,
                ),
                dtype=np.float64,
            )
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != expected.shape:
            return {"rel_err": float("inf")}

        worst = max(worst, scorers.rel_err(expected, got))

    return {"rel_err": worst}
