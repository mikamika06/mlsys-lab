import ref


def check(workdir):
    from logaudit.audit import compute_leakage_stats, parse_and_audit

    out = {
        "violations_match": 0.0,
        "leaked_tokens_match": 0.0,
        "isolation_score_match": 0.0,
    }

    logs = ref.generate_traces(seed=202)

    ref_tracker = {}
    ref_tokens = {}
    ref_violations = []

    for event in logs:
        etype = event["type"]
        tid = event["tenant_id"]
        bid = event["block_id"]
        if etype == "allocate":
            ref_tracker[bid] = tid
            ref_tokens[bid] = event["tokens"]
        elif etype == "lookup":
            owner = ref_tracker.get(bid)
            if owner and owner != tid:
                ref_violations.append({
                    "request_id": event["request_id"],
                    "tenant_id": tid,
                    "owner_tenant_id": owner,
                    "block_id": bid,
                    "tokens_leaked": len(ref_tokens.get(bid, [])),
                })

    ref_stats = {
        "total_violations": len(ref_violations),
        "total_leaked_tokens": sum(v["tokens_leaked"] for v in ref_violations),
        "unique_pair_violations": len({(v["owner_tenant_id"], v["tenant_id"]) for v in ref_violations}),
    }

    try:
        got_violations = parse_and_audit(logs)
        if got_violations == ref_violations:
            out["violations_match"] = 1.0

        got_stats = compute_leakage_stats(got_violations)
        if got_stats.get("total_leaked_tokens") == ref_stats["total_leaked_tokens"]:
            out["leaked_tokens_match"] = 1.0

        if got_stats == ref_stats:
            out["isolation_score_match"] = 1.0
        else:
            out["_note"] = f"Expected stats {ref_stats}, got {got_stats}"
    except Exception as e:
        out["_note"] = f"Error evaluating leakage: {type(e).__name__}: {e}"

    return out
