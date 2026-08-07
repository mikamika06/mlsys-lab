import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"classifications_matched": 0.0, "scheduler_stats_matched": 0.0}

    try:
        from warpanalyze.classify import classify_kernel_profile
        from warpanalyze.scheduler import compute_issue_slot_utilization
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {str(e)}"
        return out

    class_cases = ref.generate_kernel_classification_cases()
    class_passed = 0
    for stats, want_cat in class_cases:
        try:
            got_cat = classify_kernel_profile(stats)
            if got_cat == want_cat:
                class_passed += 1
            elif "_note" not in out:
                out["_note"] = f"Classification mismatch: got {got_cat}, want {want_cat}"
        except Exception as e:
            out["_note"] = f"Classify error: {type(e).__name__}: {str(e)}"
            return out

    if class_passed == len(class_cases):
        out["classifications_matched"] = 1.0

    sched_cases = ref.generate_scheduler_stats_cases()
    sched_passed = 0
    for sched_input in sched_cases:
        want_res = ref.ref_compute_issue_slot_utilization(sched_input)
        try:
            got_res = compute_issue_slot_utilization(sched_input)
        except Exception as e:
            out["_note"] = f"Scheduler error: {type(e).__name__}: {str(e)}"
            return out

        match = True
        for k, v in want_res.items():
            if k not in got_res or abs(got_res[k] - v) > 1e-5:
                match = False
                break
        if match:
            sched_passed += 1
        elif "_note" not in out:
            out["_note"] = f"Scheduler mismatch: got {got_res}, want {want_res}"

    if sched_passed == len(sched_cases):
        out["scheduler_stats_matched"] = 1.0

    return out
