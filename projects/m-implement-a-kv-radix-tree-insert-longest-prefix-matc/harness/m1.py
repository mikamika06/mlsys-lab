import ref

def check(workdir):
    from kvtree.tree import RadixTree
    out = {"tree_matches": 0.0}
    tree = RadixTree()
    tree.insert([1, 2, 3, 4], [10, 11])
    tree.insert([1, 2, 5, 6], [10, 12])
    blocks, matched = tree.match_prefix([1, 2, 3, 4, 99])
    ref_tree = ref.RefRadixTree()
    ref_tree.insert([1, 2, 3, 4], [10, 11])
    ref_tree.insert([1, 2, 5, 6], [10, 12])
    ref_blocks, ref_matched = ref_tree.match_prefix([1, 2, 3, 4, 99])
    if blocks == ref_blocks and matched == ref_matched:
        out["tree_matches"] = 1.0
    else:
        out["_note"] = f"Expected {ref_blocks}, {ref_matched}, got {blocks}, {matched}"
    return out
