import numpy as np
import ref


def check(workdir):
    from fsdpfit.shards import simulate_fsdp_shards

    out = {"shards_matched": 0.0}
    ok = True
    for i, model in enumerate(ref.TEST_MODELS):
        want = ref.simulate_fsdp_shards(model, world_size=2)
        got = simulate_fsdp_shards(model, world_size=2)
        if len(want) != len(got):
            ok = False
            break
        for r in range(len(want)):
            for k in want[r]:
                if not np.array_equal(want[r][k], got[r][k]):
                    ok = False
                    out["_note"] = f"model {i} rank {r} param {k} mismatch"
                    break
    if ok:
        out["shards_matched"] = 1.0
    return out
