import workload as W


def check(workdir):
    spec = W.trace(14, 96, 24, 2, prefix=W.SHARED)
    cfg = dict(W.CFG_BASE, prefix_cache=True, max_batch_tokens=384, num_blocks=90)
    mine, tr = W.learner(workdir, spec, cfg)
    ref, rtr = W.oracle(spec, cfg)
    n = min(len(tr), len(rtr))
    same = sum(1 for i in range(n) if tuple(tr[i].get("ids", ())) == tuple(rtr[i]["ids"]))
    denom = max(len(tr), len(rtr), 1)
    return {
        "schedule_match": same / denom,
        "steps_delta": float(abs(len(tr) - len(rtr))),
        "metrics_match": 1.0 if all(mine.get(k) == ref[k] for k in
                                    ("finished", "rejected", "prefill_tokens", "decode_tokens")) else 0.0,
    }
