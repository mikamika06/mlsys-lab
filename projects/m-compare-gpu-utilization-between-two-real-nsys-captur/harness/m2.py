import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from nsys_analyzer.allocation import compute_allocation_churn_overhead

    out = {"churn_ratio_matched": 0.0}

    ref_churn = ref.compute_allocation_churn_overhead(ref.CUDA_API_REPORT)
    got_churn = compute_allocation_churn_overhead(ref.CUDA_API_REPORT)

    if abs(ref_churn - got_churn) < 1e-4:
        out["churn_ratio_matched"] = 1.0
    else:
        out["_note"] = f"churn ratio mismatch: ref {ref_churn}, got {got_churn}"

    return out
