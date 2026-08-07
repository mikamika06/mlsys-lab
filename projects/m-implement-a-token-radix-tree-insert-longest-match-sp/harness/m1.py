import ref

def check(workdir):
    from radixtree.tree import TokenRadixTree
    tree = TokenRadixTree()
    traces = ref.get_test_traces()
    ok = 0
    total = len(traces)
    for t in traces:
        tree.insert(t)
    for t in traces:
        matched, _ = tree.longest_match(t)
        if matched == t:
            ok += 1
    return {"tree_match": float(ok >= total)}
