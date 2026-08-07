import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"scale_t_match": 0.0, "scale_h_match": 0.0, "quant_match": 0.0}
    try:
        from fp8kv.quant import simulate_e4m3, get_per_tensor_scale, get_per_head_scale
    except ImportError as e:
        out["_note"] = f"ImportError: {e}"
        return out

    x = ref.generate_fixture()

    try:
        wt = ref.get_per_tensor_scale(x)
        gt = get_per_tensor_scale(x)
        if np.isclose(wt, gt):
            out["scale_t_match"] = 1.0
    except Exception as e:
        out["_note_t"] = str(e)

    try:
        wh = ref.get_per_head_scale(x)
        gh = get_per_head_scale(x)
        if gh.shape == (1, 8, 1) and np.allclose(wh, gh):
            out["scale_h_match"] = 1.0
    except Exception as e:
        out["_note_h"] = str(e)

    try:
        wq = ref.simulate_e4m3(x, wh)
        gq = simulate_e4m3(x, wh)
        if np.allclose(wq, gq):
            out["quant_match"] = 1.0
    except Exception as e:
        out["_note_q"] = str(e)

    return out
