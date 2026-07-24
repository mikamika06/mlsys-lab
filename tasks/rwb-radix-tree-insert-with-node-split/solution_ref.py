class _Node:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}  # first_token -> [edge_tokens_tuple, child_node]


def _insert(node, seq):
    if not seq:
        return
    first = seq[0]
    if first not in node.children:
        node.children[first] = [seq, _Node()]
        return

    edge, child = node.children[first]
    i = 0
    m = min(len(edge), len(seq))
    while i < m and edge[i] == seq[i]:
        i += 1

    if i == len(edge):
        # Edge fully consumed: recurse with the remaining suffix.
        _insert(child, seq[i:])
    else:
        # Diverges partway through the edge: split.
        common = edge[:i]
        edge_rest = edge[i:]
        new_mid = _Node()
        new_mid.children[edge_rest[0]] = [edge_rest, child]
        node.children[first] = [common, new_mid]
        if i < len(seq):
            seq_rest = seq[i:]
            new_mid.children[seq_rest[0]] = [seq_rest, _Node()]
        # else: seq ends exactly at the split point, nothing more to add.


def _collect(node, path, out):
    for _first, (edge, child) in node.children.items():
        out.append((path, edge))
        _collect(child, path + edge, out)


def build_radix_tree(sequences: list) -> list:
    """Insert `sequences` in order into a compressed-prefix (radix) tree,
    splitting edges at divergence points, and return a sorted list of
    (parent_path, edge_tokens) tuples describing every edge."""
    root = _Node()
    for seq in sequences:
        _insert(root, tuple(seq))
    out = []
    _collect(root, (), out)
    return sorted(out)
