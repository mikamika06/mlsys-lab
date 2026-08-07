import os
import tempfile
import ref

def check(workdir):
    from tritonval.layout import classify_layout
    from tritonval.versioning import compute_resident_versions

    out = {"layouts_matched": 0.0, "versions_matched": 0.0}

    # 1. Check layouts
    layout_ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        for case_name, expected_reason in ref.LAYOUT_CASES:
            model_sub = os.path.join(tmpdir, case_name)
            os.makedirs(model_sub, exist_ok=True)

            if case_name == "missing_config":
                pass
            elif case_name == "missing_versions":
                with open(os.path.join(model_sub, "config.pbtxt"), "w") as f:
                    f.write('name: "test"\nbackend: "onnx"\nmax_batch_size: 1')
            elif case_name == "malformed_version":
                with open(os.path.join(model_sub, "config.pbtxt"), "w") as f:
                    f.write('name: "test"\nbackend: "onnx"\nmax_batch_size: 1')
                os.makedirs(os.path.join(model_sub, "not_a_version"))
            elif case_name == "valid_layout":
                with open(os.path.join(model_sub, "config.pbtxt"), "w") as f:
                    f.write('name: "test"\nbackend: "onnx"\nmax_batch_size: 1')
                os.makedirs(os.path.join(model_sub, "1"))
                os.makedirs(os.path.join(model_sub, "2"))

            res = classify_layout(model_sub)
            if res != expected_reason:
                layout_ok = False
                if "_note" not in out:
                    out["_note"] = f"layout {case_name}: got {res}, want {expected_reason}"
                break

    if layout_ok:
        out["layouts_matched"] = 1.0

    # 2. Check versioning policies
    versions_ok = True
    for avail, policy, expected in ref.VERSION_CASES:
        got = compute_resident_versions(avail, policy)
        if got != expected:
        # Check sorted equality just in case order differs slightly
            versions_ok = False
            if "_note" not in out:
                out["_note"] = f"versioning policy {policy} with {avail}: got {got}, want {expected}"
            break

    if versions_ok:
        out["versions_matched"] = 1.0

    return out
