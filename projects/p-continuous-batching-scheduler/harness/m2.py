import workload as W


def check(workdir):
    spec = W.trace(12, 128, 32, 3)
    cfg = dict(W.CFG_BASE, max_batch_tokens=256)
    mine, tr = W.learner(workdir, spec, cfg)
    ref, rtr = W.oracle(spec, cfg)
    over = sum(1 for r in tr
               if r.get("prefill_tokens", 0) + r.get("decode_tokens", 0) > cfg["max_batch_tokens"])
    return {
        "budget_violations": float(over),
        "prefill_tokens": float(mine.get("prefill_tokens", -1)),
        "prefill_tokens_match": 1.0 if mine.get("prefill_tokens") == ref["prefill_tokens"] else 0.0,
        "admitted_all": 1.0 if mine.get("finished", 0) + mine.get("rejected", 0) == 12 else 0.0,
    }
