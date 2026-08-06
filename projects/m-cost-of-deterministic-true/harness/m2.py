import ref
import numpy as np

def check(workdir):
    from detcost.diagnose import diagnose_divergence
    out = {"divergence_detected": 0.0}
    test_cases = [
        ([1.0, 1.1, float("nan"), 1.2], [0.1, 0.2, 0.3, 0.4]),
        ([1.0, 1.2, 1.3, 1.4], [0.1, 0.2, 0.3, 1e6]),
        ([1.0, 1.1, 1.2, 1.3], [0.1, 0.1, 0.1, 0.1])
    ]
    ok = 0
    for losses, norms in test_cases:
        want = ref.get_diagnose_result(losses, norms)
        got = diagnose_divergence(losses, norms)
        if got.get("diverged") == want.get("diverged") and got.get("step") == want.get("step"):
            ok += 1
    if ok == len(test_cases):
        out["divergence_detected"] = 1.0
    else:
        out["_note"] = f"diagnose test passed {ok}/{len(test_cases)} cases"
    return out
