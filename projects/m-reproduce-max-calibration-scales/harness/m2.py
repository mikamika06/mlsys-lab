import sys
import numpy as np
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"kl_scales_matched": 0.0, "rel_err": 1.0}

    try:
        from calib.entropy_calib import compute_entropy_scale
    except Exception as e:
        out["_note"] = f"Failed to import entropy_calib: {e}"
        return out

    tensors = ref.generate_test_tensors()
    matched = 0
    total = len(tensors)
    max_rel_err = 0.0

    try:
        for t in tensors:
            want = ref.ref_compute_entropy_scale(
                t, num_bins=512, num_quant_steps=64, max_bound=127.0
            )
            got = compute_entropy_scale(
                t, num_bins=512, num_quant_steps=64, max_bound=127.0
            )

            err = abs(got - want) / (abs(want) + 1e-12)
            max_rel_err = max(max_rel_err, err)

            if np.isclose(got, want, rtol=1e-3, atol=1e-4):
                matched += 1

        if matched == total:
            out["kl_scales_matched"] = 1.0

        out["rel_err"] = float(max_rel_err)

    except Exception as e:
        out["_note"] = f"Execution error in m2: {e}"

    return out
