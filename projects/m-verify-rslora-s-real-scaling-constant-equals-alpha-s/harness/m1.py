import ref
import numpy as np

def check(workdir):
    from adapters.verify import verify_rslora_scaling
    cases = ref.generate_test_cases()
    out = {"scaling_verified": 0.0}
    ok = 0
    for case in cases:
        r = case["rank"]
        alpha = case["alpha"]
        computed = verify_rslora_scaling(r, alpha, "rslora")
        expected = alpha / np.sqrt(r)
        if np.isclose(computed, expected, atol=1e-5):
            ok += 1
    if ok == len(cases):
        out["scaling_verified"] = 1.0
    else:
        out["_note"] = f"Verified {ok}/{len(cases)} scaling constants correctly"
    return out
