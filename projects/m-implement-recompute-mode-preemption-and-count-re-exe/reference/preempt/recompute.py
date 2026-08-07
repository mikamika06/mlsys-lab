def preempt_recompute(requests, preempt_ids):
    """Preempt specified running requests in recompute mode and count re-executed tokens."""
    p_ids = set(preempt_ids)
    updated = []
    reexecuted_tokens = 0
    for req in requests:
        r = dict(req)
        if r["req_id"] in p_ids and r.get("status") == "RUNNING":
            r["status"] = "PREEMPTED"
            r["num_blocks"] = 0
            reexecuted_tokens += r.get("prompt_len", 0) + r.get("generated_len", 0)
        updated.append(r)
    return updated, reexecuted_tokens
