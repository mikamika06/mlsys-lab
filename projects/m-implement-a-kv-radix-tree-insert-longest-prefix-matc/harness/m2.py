import ref

def check(workdir):
    from kvtree.eviction import RadixEvictor
    from kvtree.tree import RadixTree
    out = {"eviction_matches": 0.0}
    tree = RadixTree()
    evictor = RadixEvictor(tree, max_blocks=2)
    tree.insert([1, 2], [10])
    evictor.on_insert([10])
    tree.insert([3, 4], [11])
    evictor.on_insert([11])
    tree.insert([5, 6], [12])
    evictor.on_insert([12])
    if len(evictor.evicted_blocks) > 0 and 10 in evictor.evicted_blocks:
        out["eviction_matches"] = 1.0
    else:
        out["_note"] = "Eviction did not remove the oldest unpinned leaf block properly."
    return out
