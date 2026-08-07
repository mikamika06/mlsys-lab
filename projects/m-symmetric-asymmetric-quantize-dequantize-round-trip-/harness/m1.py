import numpy as np
import ref


def check(workdir):
    try:
        import quant.numerics as num
    except ImportError:
        return {"_note": "Failed to import quant.numerics"}

    out = {"asym_error_bounded": 0.0, "asym_better_for_skewed": 0.0}
    x = ref.SKEWED_DATA

    try:
        scale, zp = num.calc_scale_zp_asymmetric(np.min(x), np.max(x), 8)
        xq = num.quantize_asymmetric(x, scale, zp, 8)
        xdq = num.dequantize_asymmetric(xq, scale, zp)
        err = np.max(np.abs(x - xdq))

        if err <= scale * 0.5 + 1e-4:
            out["asym_error_bounded"] = 1.0

        scale_sym = num.calc_scale_symmetric(np.max(np.abs(x)), 8)
        xq_sym = num.quantize_symmetric(x, scale_sym, 8)
        xdq_sym = num.dequantize_symmetric(xq_sym, scale_sym)
        err_sym = np.max(np.abs(x - xdq_sym))

        if err < err_sym:
            out["asym_better_for_skewed"] = 1.0

    except NotImplementedError:
        pass

    return out
