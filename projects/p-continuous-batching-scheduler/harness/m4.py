import workload as W


def check(workdir):
    out = {}
    spec = W.trace(8, 16, 200, 0)
    mine, _ = W.learner(workdir, spec, W.CFG_TIGHT)
    ref, _ = W.oracle(spec, W.CFG_TIGHT)
    out["terminates"] = 1.0 if mine.get("steps", 10 ** 9) < 100000 else 0.0
    out["accounted"] = 1.0 if mine.get("finished", 0) + mine.get("rejected", 0) == 8 else 0.0
    out["preemptions_match"] = 1.0 if mine.get("preemptions") == ref["preemptions"] else 0.0
    out["rejected_match"] = 1.0 if mine.get("rejected") == ref["rejected"] else 0.0

    sw, _ = W.learner(workdir, spec, W.CFG_SWAP)
    rsw, _ = W.oracle(spec, W.CFG_SWAP)
    out["swap_match"] = 1.0 if (sw.get("swaps") == rsw["swaps"]
                                and sw.get("finished") == rsw["finished"]) else 0.0
    return out
