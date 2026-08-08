import ref


def check(workdir):
    from tvm_compare.latency import compute_latency_ratio

    ok = 0
    total = len(ref.MODELS)
    for m in ref.MODELS:
        want = ref.compute_latency_ratio(m["name"], m["torch"], m["tvm"]) if hasattr(ref, "compute_latency_ratio") else 0.0
        # Compute reference directly if not in ref module
        import numpy as np
        t_torch = float(np.mean(m["torch"]))
        t_tvm = float(np.mean(m["tvm"]))
        expected = t_torch / t_tvm if t_tvm > 0 else 0.0

        got = compute_latency_ratio(m["name"], m["torch"], m["tvm"])
        if abs(got - expected) < 1e-5:
            ok += 1

    out = {"latency_ratio_matched": 1.0 if ok == total else 0.0}
    return out
