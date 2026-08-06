import ref


def check(workdir):
    from loraparams.compare import audit_recorded_run, summarize_audits

    out = {
        "runs_audited": 0.0,
        "mismatches_detected": 0.0,
        "summaries_matched": 0.0,
    }
    runs = ref.get_sample_runs()
    audits = []
    for r in runs:
        try:
            res = audit_recorded_run(r)
            audits.append(res)
        except Exception as e:
            out["_note"] = f"audit_recorded_run raised {type(e).__name__}: {e}"
            return out

    if len(audits) == len(runs):
        out["runs_audited"] = 1.0

    mismatches = [a["run_id"] for a in audits if not a.get("is_valid", False)]
    if mismatches == ["run-1001", "run-1003"]:
        out["mismatches_detected"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"expected mismatches ['run-1001', 'run-1003'], got {mismatches}"

    try:
        summary = summarize_audits(audits)
        expected_summary = {
            "total_runs": 5,
            "valid_runs": 3,
            "mismatched_runs": 2,
            "mismatched_ids": ["run-1001", "run-1003"],
        }
        if summary == expected_summary:
            out["summaries_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"summary mismatch: got {summary}, want {expected_summary}"
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"summarize_audits raised {type(e).__name__}: {e}"

    return out
