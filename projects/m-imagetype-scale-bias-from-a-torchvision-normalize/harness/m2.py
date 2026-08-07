import ref
import numpy as np


def check(workdir):
    from export.verify import compute_relative_drift
    from export.size import compute_package_size_ratio

    out = {"rel_err": 1.0, "size_ratio_matched": 0.0}

    torch_out, coreml_out = ref.generate_cnn_outputs()
    try:
        drift = compute_relative_drift(torch_out, coreml_out)
        out["rel_err"] = float(drift)
    except Exception as e:
        out["_note"] = f"compute_relative_drift raised {type(e).__name__}: {e}"
        return out

    fp32_spec, fp16_spec = ref.generate_package_specs()

    fp32_bytes = sum(fp32_spec["weights"].values()) + fp32_spec["metadata_bytes"]
    fp16_bytes = sum(fp16_spec["weights"].values()) + fp16_spec["metadata_bytes"]
    expected_ratio = float(fp32_bytes / fp16_bytes)

    try:
        got_ratio = compute_package_size_ratio(fp32_spec, fp16_spec)
        if np.isclose(got_ratio, expected_ratio, rtol=1e-4):
            out["size_ratio_matched"] = 1.0
        else:
            out["_note"] = f"got size ratio {got_ratio}, expected {expected_ratio}"
    except Exception as e:
        out["_note"] = f"compute_package_size_ratio raised {type(e).__name__}: {e}"

    return out
