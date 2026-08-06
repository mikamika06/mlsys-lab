import ref

def check(workdir):
    from batcher.policy import compute_rejected_requests
    out = {"rejections_matched": 0.0}
    ok = 0
    for sc in ref.SCENARIOS:
        got = compute_rejected_requests(
            sc["queue_size"], sc["max_queue_size"], sc["incoming_count"], sc["policy"]
        )
        if got == sc["want_rejected"]:
            ok += 1
    out["rejections_matched"] = float(ok)
    return out
