import ref


def check(workdir):
    from zeromem.optimizer import optimizer_state_memory
    out = {"optimizer_matched": 0.0, "total": 0.0}
    ok = 0
    total = 0
    for p in ref.PARAMS_LIST:
        for w in ref.WORLD_SIZES:
            for sharded in (True, False):
                total += 1
                want = 12 * p // w if sharded else 12 * p
                got = optimizer_state_memory(p, w, sharded=sharded)
                if got == want:
                    ok += 1
    out["optimizer_matched"] = float(ok)
    out["total"] = float(total)
    return out
