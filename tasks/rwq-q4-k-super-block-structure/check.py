import numpy as np


def _oracle_quantize(x):
    x = np.asarray(x, dtype=np.float32)
    rows, cols = x.shape
    n_sb = cols // 256

    codes = np.zeros((rows, cols // 2), dtype=np.uint8)
    sub_scales = np.zeros((rows, n_sb, 8), dtype=np.uint8)
    sub_mins = np.zeros((rows, n_sb, 8), dtype=np.uint8)
    d = np.zeros((rows, n_sb), dtype=np.float16)
    dmin = np.zeros((rows, n_sb), dtype=np.float16)

    for r in range(rows):
        for sb in range(n_sb):
            block = x[r, sb * 256:(sb + 1) * 256]
            ss = []
            mm = []
            for i in range(8):
                sub = block[i * 32:(i + 1) * 32]
                mn = float(np.min(sub))
                mx = float(np.max(sub))
                ss.append((mx - mn) / 63.0)
                mm.append(-mn / 63.0)

            ds = max(ss)
            dm = max(mm)
            d[r, sb] = np.float16(ds)
            dmin[r, sb] = np.float16(dm)

            for i in range(8):
                sc = 0 if ds == 0 else int(np.clip(round(ss[i] / ds * 63), 0, 63))
                mc = 0 if dm == 0 else int(np.clip(round(mm[i] / dm * 63), 0, 63))
                sub_scales[r, sb, i] = sc
                sub_mins[r, sb, i] = mc

                sub = block[i * 32:(i + 1) * 32]
                step = float(d[r, sb]) * sc
                off = float(dmin[r, sb]) * mc
                q = (
                    np.zeros(32, dtype=np.uint8)
                    if step == 0
                    else np.clip(np.round((sub + off) / step), 0, 15).astype(np.uint8)
                )
                base = sb * 128 + i * 16
                codes[r, base:base + 16] = q[::2] | (q[1::2] << 4)

    return codes, sub_scales, sub_mins, d, dmin


def _oracle_dequant(codes, sub_scales, sub_mins, d, dmin):
    codes = np.asarray(codes, dtype=np.uint8)
    rows, packed = codes.shape
    n_sb = packed // 128
    out = np.zeros((rows, n_sb * 256), dtype=np.float32)

    for r in range(rows):
        for sb in range(n_sb):
            dv = float(d[r, sb])
            dmv = float(dmin[r, sb])
            for i in range(8):
                sc = int(sub_scales[r, sb, i])
                mc = int(sub_mins[r, sb, i])
                qbytes = codes[r, sb * 128 + i * 16:sb * 128 + i * 16 + 16]
                q = np.empty(32, dtype=np.uint8)
                q[::2] = qbytes & 15
                q[1::2] = qbytes >> 4
                out[r, sb * 256 + i * 32:sb * 256 + i * 32 + 32] = (
                    dv * sc * q.astype(np.float32) - dmv * mc
                )
    return out


def _cases():
    rng = np.random.default_rng(42)
    rows = []
    rows.append(np.sin(np.arange(256, dtype=np.float32) / 11.0) * 4)
    rows.append(np.linspace(-8, 7, 256, dtype=np.float32))
    rows.append(
        np.concatenate(
            [
                np.full(64, -3, dtype=np.float32),
                np.linspace(-1, 5, 64, dtype=np.float32),
                np.linspace(5, -6, 64, dtype=np.float32),
                np.zeros(64, dtype=np.float32),
            ]
        )
    )
    rows.append((rng.normal(size=512) * 2.0).astype(np.float32))  # two super-blocks
    return np.stack(rows[:3]).astype(np.float32), rows[3][None, :].astype(np.float32)


def grade(sol, fx) -> dict:
    x1, x2 = _cases()

    worst_rel = 0.0
    codes_exact = 1.0

    for x in (x1, x2):
        ref_codes, ref_ss, ref_sm, ref_d, ref_dm = _oracle_quantize(x)
        ref_recon = _oracle_dequant(ref_codes, ref_ss, ref_sm, ref_d, ref_dm)

        try:
            codes, sub_scales, sub_mins, d, dmin = sol.q4k_quantize_superblock(
                np.array(x, copy=True)
            )
            recon = sol.q4k_dequantize_superblock(codes, sub_scales, sub_mins, d, dmin)
        except Exception:
            return {"rel_err": float("inf"), "codes_exact": 0.0}

        codes = np.asarray(codes, dtype=np.uint8)
        sub_scales = np.asarray(sub_scales)
        sub_mins = np.asarray(sub_mins)
        recon = np.asarray(recon, dtype=np.float32)

        if (
            codes.shape != ref_codes.shape
            or sub_scales.shape != ref_ss.shape
            or sub_mins.shape != ref_sm.shape
            or recon.shape != ref_recon.shape
        ):
            return {"rel_err": float("inf"), "codes_exact": 0.0}

        if not np.array_equal(codes, ref_codes):
            codes_exact = 0.0
        if np.any(sub_scales.astype(np.int64) < 0) or np.any(sub_scales.astype(np.int64) > 63):
            codes_exact = 0.0
        if np.any(sub_mins.astype(np.int64) < 0) or np.any(sub_mins.astype(np.int64) > 63):
            codes_exact = 0.0
        if not np.array_equal(sub_scales.astype(np.int64), ref_ss.astype(np.int64)):
            codes_exact = 0.0
        if not np.array_equal(sub_mins.astype(np.int64), ref_sm.astype(np.int64)):
            codes_exact = 0.0

        if not np.all(np.isfinite(recon)):
            return {"rel_err": float("inf"), "codes_exact": codes_exact}

        rel = np.linalg.norm((recon - ref_recon).ravel()) / (
            np.linalg.norm(ref_recon.ravel()) + 1e-12
        )
        worst_rel = max(worst_rel, float(rel))

    return {"rel_err": worst_rel, "codes_exact": codes_exact}
