import struct
from fractions import Fraction

import numpy as np

from mlsys import scorers


def _oracle_fields(x: float) -> tuple[int, int, int]:
    """Real CPython oracle: the actual binary64 byte pattern via struct."""
    bits = struct.unpack("<Q", struct.pack("<d", float(x)))[0]
    return (bits >> 63) & 1, (bits >> 52) & 0x7FF, bits & ((1 << 52) - 1)


def _fail():
    return {
        "fields_exact_fraction": 0.0,
        "ratio_exact_fraction": 0.0,
        "max_rel_err": float("inf"),
        "roundtrip_max_abs_err": float("inf"),
    }


def grade(sol, fx) -> dict:
    xs = [float(v) for v in np.asarray(fx["values"], dtype=np.float64).ravel()]

    field_hits = 0
    ratio_hits = 0
    max_rel = 0.0
    recon = []

    for x in xs:
        # ---- oracle, computed here, never hardcoded ----
        ref_fields = _oracle_fields(x)
        ref_frac = Fraction(x)                       # CPython exact float -> rational

        try:
            got_fields = sol.float_fields(x)
            got_ratio = sol.exact_ratio(x)
            s, e, m = (int(got_fields[0]), int(got_fields[1]), int(got_fields[2]))
            num, den = int(got_ratio[0]), int(got_ratio[1])
        except Exception:
            return _fail()

        if (s, e, m) == ref_fields:
            field_hits += 1

        if den <= 0:
            return _fail()

        got_frac = Fraction(num, den)
        # lowest terms + positive denominator is exactly what Fraction normalises to
        if num == ref_frac.numerator and den == ref_frac.denominator:
            ratio_hits += 1

        # exact rational relative error, then a single float conversion at the end
        if ref_frac == 0:
            rel = float(abs(got_frac))
        else:
            rel = float(abs(got_frac - ref_frac) / abs(ref_frac))
        max_rel = max(max_rel, rel)

        recon.append(float(got_frac))

    n = len(xs)
    return {
        "fields_exact_fraction": field_hits / n,
        "ratio_exact_fraction": ratio_hits / n,
        "max_rel_err": float(max_rel),
        "roundtrip_max_abs_err": scorers.max_abs_err(
            np.array(xs, dtype=np.float64), np.array(recon, dtype=np.float64)
        ),
    }
