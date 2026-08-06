import ref


def check(workdir):
    from triage.exporter import run_draft_export
    from triage.report import generate_triage_report

    out = {"report_structure_valid": 0.0, "scores_matched": 0.0}

    results = [run_draft_export(m) for m in ref.TEST_MODELS]
    ref_results = [ref.run_draft_export(m) for m in ref.TEST_MODELS]

    got_report = generate_triage_report(results)
    want_report = ref.generate_triage_report(ref_results)

    required_keys = {
        "total_models",
        "successful_exports",
        "failed_exports",
        "total_issues",
        "issue_counts",
        "priority_score",
        "status"
    }

    if set(got_report.keys()) == required_keys:
        out["report_structure_valid"] = 1.0
    else:
        out["_note"] = f"Missing required keys in report: {required_keys - set(got_report.keys())}"
        return out

    if (
        got_report["priority_score"] == want_report["priority_score"]
        and got_report["status"] == want_report["status"]
        and got_report["issue_counts"] == want_report["issue_counts"]
    ):
        out["scores_matched"] = 1.0
    else:
        out["_note"] = f"Report output mismatch: got {got_report}, want {want_report}"

    return out
