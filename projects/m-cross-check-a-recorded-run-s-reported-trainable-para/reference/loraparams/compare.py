from loraparams.formula import calculate_trainable_params


def audit_recorded_run(run_record: dict) -> dict:
    """Audits a single run record against the expected parameter formula."""
    run_id = run_record.get("run_id", "unknown")
    reported = run_record.get("reported_trainable_params", 0)
    expected_breakdown = calculate_trainable_params(
        run_record.get("model_config", {}),
        run_record.get("lora_config", {}),
    )
    expected_total = expected_breakdown["total_trainable_params"]
    delta = reported - expected_total
    is_valid = delta == 0

    return {
        "run_id": run_id,
        "is_valid": is_valid,
        "status": "MATCH" if is_valid else "MISMATCH",
        "expected_trainable_params": expected_total,
        "reported_trainable_params": reported,
        "delta": delta,
        "breakdown": expected_breakdown,
    }


def summarize_audits(audit_results: list) -> dict:
    """Summarizes audit results across multiple run records."""
    total_runs = len(audit_results)
    valid_runs = sum(1 for a in audit_results if a.get("is_valid", False))
    mismatched_runs = total_runs - valid_runs
    mismatched_ids = [a["run_id"] for a in audit_results if not a.get("is_valid", False)]

    return {
        "total_runs": total_runs,
        "valid_runs": valid_runs,
        "mismatched_runs": mismatched_runs,
        "mismatched_ids": sorted(mismatched_ids),
    }
