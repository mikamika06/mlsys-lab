import numpy as np
import ref


def check(workdir):
    from bench.agreement import evaluate_mac_runs

    records = ref.generate_mac_records()
    want = ref.build_mac_agreements(records)

    out = {"mac_agreements_matched": 0.0}
    try:
        got = evaluate_mac_runs(records)
    except Exception as e:
        out["_note"] = f"evaluate_mac_runs failed: {type(e).__name__}: {str(e)[:120]}"
        return out

    if len(got) != len(want):
        out["_note"] = f"Expected {len(want)} results, got {len(got)}"
        return out

    ok = True
    for g, w in zip(got, want):
        if g["model_id"] != w["model_id"] or g["agreed"] != w["agreed"]:
            ok = False
            break
        if not np.isclose(g["max_rel_err"], w["max_rel_err"], rtol=1e-4, atol=1e-6):
            ok = False
            break
        if not np.isclose(g["ort_latency_ms"], w["ort_latency_ms"], rtol=1e-4, atol=1e-6):
            ok = False
            break
        if not np.isclose(g["ov_latency_ms"], w["ov_latency_ms"], rtol=1e-4, atol=1e-6):
            ok = False
            break
        if not np.isclose(g["latency_ratio_ort_over_ov"], w["latency_ratio_ort_over_ov"], rtol=1e-4, atol=1e-6):
            ok = False
            break

    if ok:
        out["mac_agreements_matched"] = 1.0
    else:
        out["_note"] = f"Mismatch in Mac evaluation results. Sample got: {got[0]}, want: {want[0]}"

    return out
