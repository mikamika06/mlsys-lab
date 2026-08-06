import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.group_int4 import classify_elements

    test_cases = ref.generate_test_cases()
    out = {"classification_matched": 0.0, "error_breakdown_matched": 0.0}

    class_ok = True
    breakdown_ok = True

    for i, case in enumerate(test_cases):
        x = case["x"]
        g = case["group_size"]
        want = ref.classify_elements(x, g)
        try:
            got = classify_elements(x, g)

            mask_match = np.array_equal(got["clamped_mask"], want["clamped_mask"])
            count_match = (got["clamped_count"] == want["clamped_count"]) and (got["in_range_count"] == want["in_range_count"])
            ratio_match = np.isclose(got["clamped_ratio"], want["clamped_ratio"], atol=1e-6)

            if not (mask_match and count_match and ratio_match):
                class_ok = False
                if "_note" not in out:
                    out["_note"] = f"classification mismatch on case {i}"

            c_mse_match = np.isclose(got["clamped_mse_contrib"], want["clamped_mse_contrib"], atol=1e-6)
            r_mse_match = np.isclose(got["in_range_mse_contrib"], want["in_range_mse_contrib"], atol=1e-6)

            if not (c_mse_match and r_mse_match):
                breakdown_ok = False
                if "_note" not in out:
                    out["_note"] = f"error breakdown mismatch on case {i}"

        except Exception as e:
            class_ok = False
            breakdown_ok = False
            if "_note" not in out:
                out["_note"] = f"case {i} raised exception: {type(e).__name__}: {str(e)[:100]}"

    out["classification_matched"] = 1.0 if class_ok else 0.0
    out["error_breakdown_matched"] = 1.0 if breakdown_ok else 0.0
    return out
