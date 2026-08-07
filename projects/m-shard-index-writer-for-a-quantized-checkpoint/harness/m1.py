import ref


def check(workdir):
    from qindex.writer import build_shard_index

    out = {"indices_matched": 0.0, "total": float(len(ref.CHECKPOINTS))}
    ok = 0
    for i, cp in enumerate([ref.CHECKPOINTS[0], ref.CHECKPOINTS[1], {"shard_name": ref.CHECKPOINTS[2]["shard_name"], "tensors": ref.CHECKPOINTS[2]["tensors"]}]):
        want = ref.build_index([cp])
        got = build_shard_index([cp])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"checkpoint {i} mismatch: got {got}, want {want}"
    out["indices_matched"] = float(ok)
    return out
