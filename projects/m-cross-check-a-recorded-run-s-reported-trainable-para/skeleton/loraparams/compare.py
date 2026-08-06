def audit_recorded_run(run_record: dict) -> dict:
    """Audits a single run record against the expected parameter formula."""
    raise NotImplementedError


def summarize_audits(audit_results: list) -> dict:
    """Summarizes audit results across multiple run records."""
    raise NotImplementedError
