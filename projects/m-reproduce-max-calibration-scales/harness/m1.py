import sys
import numpy as np
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"max_scales_matched": 0.0, "rel_err": 1.0}

    try:
        from calib.max_calib import MaxCalibrator, compute_max_scale
    except Exception as e:
        out["_note"] = f"Failed to import max_calib: {e}"
        return out

    tensors = ref.generate_test_tensors()
    matched = 0
    total = len(tensors)
    max_rel_err = 0.0

    try:
        for t in tensors:
            want = ref.ref_compute_max_scale(t, max_bound=127.0)
            got = compute_max_scale(t, max_bound=127.0)

            err = abs(got - want) / (abs(want) + 1e-12)
            max_rel_err = max(max_rel_err, err)

            if np.isclose(got, want, rtol=1e-5, atol=1e-6):
                matched += 1

        calibrator = MaxCalibrator(max_bound=127.0)
        for t in tensors:
            calibrator.collect(t)
        calib_got = calibrator.compute_scale()
        calib_want = ref.ref_compute_max_scale(
            np.concatenate([x.flatten() for x in tensors]), max_bound=127.0
        )

        calib_err = abs(calib_got - calib_want) / (abs(calib_want) + 1e-12)
        max_rel_err = max(max_rel_err, calib_err)

        if np.isclose(calib_got, calib_want, rtol=1e-5, atol=1e-6):
            if matched == total:
                out["max_scales_matched"] = 1.0

        out["rel_err"] = float(max_rel_err)

    except Exception as e:
        out["_note"] = f"Execution error in m1: {e}"

    return out
