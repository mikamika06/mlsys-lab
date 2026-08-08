def compute_success_rate(records):
    """Compute conversion success percentage from a list of records."""
    if not records:
        return 0.0
    successes = sum(1 for r in records if r.get("status") == "success")
    return float(successes) / float(len(records))
