import ref


def check(workdir):
    from evict.parser import reconstruct_eviction_order

    snapshots, models = ref.generate_snapshots()
    want = ref.reconstruct_eviction_order(snapshots)
    got = reconstruct_eviction_order(snapshots)
    out = {"order_matched": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"got eviction order {got}, expected {want}"
    return out
