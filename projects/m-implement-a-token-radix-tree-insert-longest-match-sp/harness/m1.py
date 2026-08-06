import ref

def check(workdir):
    from radix.tree import TokenRadixTree
    tree = TokenRadixTree()
    for trace in ref.TRACES:
        tree.insert(trace, value=True)
    matched, _ = tree.longest_match([1, 2, 3, 4, 5, 6, 99])
    if matched == [1, 2, 3, 4, 5, 6]:
        return {"tree_matched": 1.0}
    return {"tree_matched": 0.0, "_note": f"got matched {matched}"}
