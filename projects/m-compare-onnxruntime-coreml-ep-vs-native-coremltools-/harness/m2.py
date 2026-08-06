import ref
import numpy as np


def check(workdir):
    from edgecomp.fallback import analyze_fallback
    models = ref.generate_fixtures()
    matched = 0
    total = len(models)
    for model in models:
        got = analyze_fallback(model)
        ep_ref = ref.compute_coreml_ep_execution(model)
        native_ref = ref.compute_native_execution(model)
        want_overhead = ep_ref["latency"] - native_ref["latency"]
        want_total = ep_ref["latency"]
        want_frac = want_overhead / (want_total + 1e-9)
        if (
            isinstance(got, dict)
            and "fallback_overhead" in got
            and "fallback_fraction" in got
            and np.isclose(got["fallback_overhead"], want_overhead, rtol=1e-3)
            and np.isclose(got["fallback_fraction"], want_frac, rtol=1e-3)
        ):
            matched += 1
    out = {"fallback_cost_matched": 1.0 if matched == total else 0.0}
    if matched < total:
        out["_note"] = f"Matched {matched}/{total} models in fallback cost analysis"
    return out
