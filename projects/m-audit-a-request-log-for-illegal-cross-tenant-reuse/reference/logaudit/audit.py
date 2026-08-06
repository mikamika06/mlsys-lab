from logaudit.tracker import CacheTracker


def parse_and_audit(logs, block_size=16):
    """Audits request logs for cross-tenant KV cache reuses."""
    tracker = CacheTracker(block_size=block_size)
    violations = []
    for event in logs:
        v = tracker.process_event(event)
        if v is not None:
            violations.append(v)
    return violations


def compute_leakage_stats(violations):
    """Calculates total leaked tokens and isolation score."""
    total_violations = len(violations)
    total_leaked_tokens = sum(v["tokens_leaked"] for v in violations)
    unique_offenders = len({(v["owner_tenant_id"], v["tenant_id"]) for v in violations})
    return {
        "total_violations": total_violations,
        "total_leaked_tokens": total_leaked_tokens,
        "unique_pair_violations": unique_offenders,
    }
