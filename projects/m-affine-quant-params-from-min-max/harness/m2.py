import sys
import numpy as np
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from quantizer.static_quant import CalibrationDataReader, calibrate_static_params
    from quantizer.report import compare_dynamic_vs_static

    batches = ref.generate_calibration_dataset(seed=42)
    reader = CalibrationDataReader(batches)

    out = {"calib_matched": 0.0, "size_ratio": 1.0}

    try:
        params = calibrate_static_params(reader)
    except Exception as e:
        out["_note"] = f"calibrate_static_params raised {type(e).__name__}: {e}"
        return out

    stats_want = {}
    for batch in batches:
        for k, arr in batch.items():
            if k not in stats_want:
                stats_want[k] = [float(np.min(arr)), float(np.max(arr))]
            else:
                stats_want[k][0] = min(stats_want[k][0], float(np.min(arr)))
                stats_want[k][1] = max(stats_want[k][1], float(np.max(arr)))

    want_params = {}
    for k, (g_min, g_max) in stats_want.items():
        want_params[k] = ref.calc_affine_params_ref(g_min, g_max, 0, 255)

    all_matched = True
    for k, (w_s, w_zp) in want_params.items():
        if k not in params:
            all_matched = False
            out["_note"] = f"missing tensor {k} in params"
            break
        g_s, g_zp = params[k]
        if abs(g_s - w_s) > 1e-5 or g_zp != w_zp:
            all_matched = False
            out["_note"] = f"tensor {k}: got ({g_s}, {g_zp}), want ({w_s}, {w_zp})"
            break

    if all_matched:
        out["calib_matched"] = 1.0

    try:
        rep = compare_dynamic_vs_static(batches, params)
        ratios = [v["size_ratio"] for v in rep.values() if "size_ratio" in v]
        if ratios:
            out["size_ratio"] = float(np.mean(ratios))
    except Exception as e:
        out["_note"] = f"compare_dynamic_vs_static raised {type(e).__name__}: {e}"

    return out
