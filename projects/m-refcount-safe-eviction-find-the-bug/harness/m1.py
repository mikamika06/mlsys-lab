import ref


def check(workdir):
    from radixkv.eviction import simulate_eviction
    out = {"eviction_matched": 0.0}
    ok = 0
    tests = [
        ({"nodes": [{"id": 1, "parent": None, "refcount": 1, "children": [2]}, {"id": 2, "parent": 1, "refcount": 1, "children": []}]}, 2),
        ({"nodes": [{"id": 1, "parent": None, "refcount": 2, "children": [2]}, {"id": 2, "parent": 1, "refcount": 1, "children": []}]}, 2),
    ]
    for tree, tid in tests:
        want = ref.simulate_eviction(tree, tid)
        try:
            got = simulate_eviction(tree, tid)
            if sorted(got) == want:
                ok += 1
        except Exception:
            pass
    if ok == len(tests):
        out["eviction_matched"] = 1.0
    return out
