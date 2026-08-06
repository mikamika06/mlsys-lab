import ref
import numpy as np


def check(workdir):
    from edgecomp.runner import compare_runtimes
    models = ref.generate_fixtures()
    matched = 0
    total = len(models)
    for model in models:
        got = compare_runtimes(model)
        native_ref = ref.compute_native_execution(model)
        ep_ref = ref.compute_coreml_ep_execution(model)
        want_ratio = ep_ref["latency"] / (native_ref["latency"] + 1e-9)
        if (
            isinstance(got, dict)
            and "latency_ratio" in got
            and "outputs_match" in got
            and got["outputs_match"]
            and np.isclose(got["latency_ratio"], want_ratio, rtol=1e-3)
        ):
            matched += 1
    out = {"latency_ratio_matched": 1.0 if matched == total else 0.0}
    if matched < total:
        out["_note"] = f"Matched {matched}/{total} models in runtime comparison"
    return out
