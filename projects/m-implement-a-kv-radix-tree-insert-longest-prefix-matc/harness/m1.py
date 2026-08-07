import ref
from kvradix.radix import RadixTree


def check(workdir):
    out = {"match_accuracy": 0.0, "split_correctness": 0.0}

    tree = RadixTree()
    tree.insert([1, 2, 3, 4, 5, 6], value="seq1")

    matched_len, node, rem = tree.match_prefix([1, 2, 3, 4, 7, 8])
    if matched_len == 4 and rem == [7, 8]:
        out["match_accuracy"] = 0.5

    matched_len2, node2, rem2 = tree.match_prefix([1, 2, 3, 4, 5, 6])
    if matched_len2 == 6 and rem2 == []:
        out["match_accuracy"] += 0.5

    tree.insert([1, 2, 3, 7, 8], value="seq2")
    if len(tree.root.children) == 1:
        prefix_node = next(iter(tree.root.children.values()))
        if prefix_node.key == [1, 2, 3] and len(prefix_node.children) == 2:
            out["split_correctness"] = 1.0
        else:
            out["_note"] = f"Expected prefix key [1, 2, 3], got {prefix_node.key}"
    else:
        out["_note"] = f"Root should have 1 child after split, got {len(tree.root.children)}"

    return out
