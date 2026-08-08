import numpy as np
import ref

def check(workdir):
    out = {"view_shapes": 0.0, "qparams_match": 0.0, "dequant_err_bounded": 0.0}
    try:
        from quant.core import get_view, restore_view, calc_qparams, apply_quant, apply_dequant
    except ImportError:
        out["_note"] = "Failed to import from quant.core"
        return out

    w = ref.FIXTURE_W[:32, :64]
    try:
        v_t = get_view(w, "tensor")
        v_a0 = get_view(w, "axis_0")
        v_a1 = get_view(w, "axis_1")
        v_g = get_view(w, "group", 16)

        # Check round-trip restoration shape matching
        rt_a1 = restore_view(v_a1, w.shape, "axis_1")

        if v_t.shape == (1, 2048) and v_a0.shape == (32, 64) and v_a1.shape == (64, 32) and v_g.shape == (128, 16) and rt_a1.shape == (32, 64):
            out["view_shapes"] = 1.0
    except Exception as e:
        out["_note"] = f"View shapes failed: {type(e).__name__}"
        return out

    try:
        s_sym, z_sym = calc_qparams(v_a0, symmetric=True)
        s_asym, z_asym = calc_qparams(v_a0, symmetric=False)

        r_s_sym, r_z_sym = ref.calc_qparams(v_a0, symmetric=True)
        r_s_asym, r_z_asym = ref.calc_qparams(v_a0, symmetric=False)

        if np.allclose(s_sym, r_s_sym) and np.array_equal(z_asym, r_z_asym):
            out["qparams_match"] = 1.0
    except Exception as e:
        out["_note"] = f"qparams matching failed: {type(e).__name__}"
        return out

    try:
        q = apply_quant(v_a0, r_s_sym, r_z_sym, symmetric=True)
        deq = apply_dequant(q, r_s_sym, r_z_sym)
        r_q = ref.apply_quant(v_a0, r_s_sym, r_z_sym, symmetric=True)
        r_deq = ref.apply_dequant(r_q, r_s_sym, r_z_sym)

        if np.max(np.abs(deq - r_deq)) < 1e-5:
            out["dequant_err_bounded"] = 1.0
    except Exception as e:
        out["_note"] = f"Dequant bounded test failed: {type(e).__name__}"

    return out
