import numpy as np
import ref


def check(workdir):
    try:
        import quant.numerics as num
    except ImportError:
        return {"_note": "Failed to import quant.numerics"}

    out = {"per_channel_better": 0.0, "fused_scale_correct": 0.0}
    w = ref.WEIGHTS

    try:
        scales_pc = num.per_channel_weight_scales(w, 8)
        wq_pc = num.quantize_symmetric(w, scales_pc, 8)
        wdq_pc = num.dequantize_symmetric(wq_pc, scales_pc)
        err_pc = np.max(np.abs(w - wdq_pc))

        scale_pt = num.calc_scale_symmetric(np.max(np.abs(w)), 8)
        wq_pt = num.quantize_symmetric(w, scale_pt, 8)
        wdq_pt = num.dequantize_symmetric(wq_pt, scale_pt)
        err_pt = np.max(np.abs(w - wdq_pt))

        if err_pc < err_pt:
            out["per_channel_better"] = 1.0

        s_in = 0.5
        s_out = 0.1
        fused = num.fused_requantize_scale(s_in, scales_pc, s_out)
        ref_fused = (s_in * scales_pc) / s_out

        if np.allclose(fused, ref_fused):
            out["fused_scale_correct"] = 1.0

    except NotImplementedError:
        pass

    return out
