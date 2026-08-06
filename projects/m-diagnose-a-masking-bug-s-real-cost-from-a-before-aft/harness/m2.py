import ref
from ncu_diag.parser import parse_ncu_summary
from ncu_diag.analysis import compute_masking_cost
from reference.analysis import compute_masking_cost as ref_analysis
from reference.parser import parse_ncu_summary as ref_parse


def check(workdir):
    out = {"cost_diff_match": 0.0}
    try:
        b_metrics = ref_parse(ref.BEFORE_CSV)
        a_metrics = ref_parse(ref.AFTER_CSV)
        got = compute_masking_cost(b_metrics, a_metrics)
        want = ref_analysis(b_metrics, a_metrics)

        matches = True
        for k in want:
            if abs(got.get(k, 0) - want[k]) > 1e-5:
                matches = False
                out["_note"] = f"Key {k} mismatch: got {got.get(k)}, want {want[k]}"
                break
        if matches:
            out["cost_diff_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Exception during analysis: {type(e).__name__}: {str(e)[:100]}"
    return out
