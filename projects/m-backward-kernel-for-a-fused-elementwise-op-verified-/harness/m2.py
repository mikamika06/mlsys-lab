import numpy as np
import ref


def check(workdir):
    from fused_grad.atomic_analysis import analyze_determinism, classify_atomic_requirement

    out = {"classification_correct": 0.0, "determinism_analysis_correct": 0.0}
    cases = ref.generate_test_cases()

    class_ok = True
    for i, c in enumerate(cases):
        got_req = classify_atomic_requirement(c["index_map"])
        want_req = c["overlaps"]
        if got_req != want_req:
            class_ok = False
            out["_note"] = f"Case {i}: expected requires_atomic={want_req}, got {got_req}"
            break

    if class_ok:
        out["classification_correct"] = 1.0

    det_ok = True
    for i, c in enumerate(cases):
        res = analyze_determinism(c["x"], c["index_map"], c["grad_output"], num_runs=5)
        if not isinstance(res, dict):
            det_ok = False
            out["_note"] = f"Case {i}: analyze_determinism did not return a dict"
            break

        keys = ["requires_atomic", "atomic_is_deterministic", "non_atomic_is_deterministic", "non_atomic_max_error"]
        if any(k not in res for k in keys):
            det_ok = False
            out["_note"] = f"Case {i}: analyze_determinism result missing keys"
            break

        if res["requires_atomic"] != c["overlaps"]:
            det_ok = False
            out["_note"] = f"Case {i}: requires_atomic key mismatch"
            break

        if not res["atomic_is_deterministic"]:
            det_ok = False
            out["_note"] = f"Case {i}: atomic execution should be deterministic"
            break

        if c["overlaps"]:
            if res["non_atomic_is_deterministic"]:
                det_ok = False
                out["_note"] = f"Case {i}: non-atomic execution with overlaps should be non-deterministic"
                break
            if res["non_atomic_max_error"] <= 1e-6:
                det_ok = False
                out["_note"] = f"Case {i}: non-atomic execution with overlaps should have non-zero max error"
                break
        else:
            if not res["non_atomic_is_deterministic"]:
                det_ok = False
                out["_note"] = f"Case {i}: non-atomic execution without overlaps should be deterministic"
                break
            if res["non_atomic_max_error"] > 1e-6:
                det_ok = False
                out["_note"] = f"Case {i}: non-atomic execution without overlaps should have zero max error"
                break

    if det_ok:
        out["determinism_analysis_correct"] = 1.0

    return out
