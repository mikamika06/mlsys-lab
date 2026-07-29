import workload as W

KEYS = ("finished", "steps", "prefill_tokens", "decode_tokens")


def check(workdir):
    out = {"cases_matched": 0.0, "cases": 3.0}
    ok = 0
    for spec, cfg in ((W.trace(12, 128, 32, 3), W.CFG_BASE),
                      (W.trace(20, 64, 16, 1), W.CFG_BASE),
                      (W.trace(6, 256, 8, 5), dict(W.CFG_BASE, max_batch_tokens=512))):
        mine, _ = W.learner(workdir, spec, cfg)
        ref, _ = W.oracle(spec, cfg)
        if all(mine.get(k) == ref[k] for k in KEYS):
            ok += 1
        else:
            out["_note"] = "; ".join(f"{k}: {mine.get(k)} != {ref[k]}"
                                     for k in KEYS if mine.get(k) != ref[k])
    out["cases_matched"] = float(ok)
    return out
