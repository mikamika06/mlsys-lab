import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from nsys_analyzer.utilization import compare_batch_utilizations, compute_gpu_utilization

    out = {"utilization_matched": 0.0, "best_batch_index_matched": 0.0}

    ref_res = ref.compare_batch_utilizations(ref.CAPTURES)
    got_res = compare_batch_utilizations(ref.CAPTURES)

    ref_utils = ref_res["utilizations"]
    got_utils = got_res.get("utilizations", [])

    if len(ref_utils) == len(got_utils):
        match_count = 0
        for ru, gu in zip(ref_utils, got_utils):
            if abs(ru - gu) < 1e-4:
                match_count += 1
        if match_count == len(ref_utils):
            out["utilization_matched"] = 1.0
        else:
            out["_note"] = f"utilizations mismatch: ref {ref_utils}, got {got_utils}"
    else:
        out["_note"] = f"utilization count mismatch: ref {len(ref_utils)}, got {len(got_utils)}"

    if got_res.get("argmin_index") == ref_res.get("argmin_index"):
        out["best_batch_index_matched"] = 1.0
    else:
        out["_note"] = f"argmin_index mismatch: ref {ref_res.get('argmin_index')}, got {got_res.get('argmin_index')}"

    return out
