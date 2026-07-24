import numpy as np


class _Node:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}


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
        _insert(child, seq[i:])
    else:
        common = edge[:i]
        edge_rest = edge[i:]
        new_mid = _Node()
        new_mid.children[edge_rest[0]] = [edge_rest, child]
        node.children[first] = [common, new_mid]
        if i < len(seq):
            seq_rest = seq[i:]
            new_mid.children[seq_rest[0]] = [seq_rest, _Node()]


def _collect(node, path, out):
    for _first, (edge, child) in node.children.items():
        out.append((path, edge))
        _collect(child, path + edge, out)


def _oracle_build(sequences):
    root = _Node()
    for s in sequences:
        _insert(root, tuple(s))
    out = []
    _collect(root, (), out)
    return sorted(out)


def grade(sol, fx) -> dict:
    seqs = np.asarray(fx["seqs"], dtype=np.int64)
    seq_lens = np.asarray(fx["seq_lens"], dtype=np.int64)
    run_id = np.asarray(fx["run_id"], dtype=np.int64)

    runs = {}
    for i in range(seqs.shape[0]):
        r = int(run_id[i])
        L = int(seq_lens[i])
        seq = [int(t) for t in seqs[i, :L]]
        runs.setdefault(r, []).append(seq)

    ok = 1.0
    for r in sorted(runs):
        sequences = runs[r]
        expected = _oracle_build(sequences)
        try:
            got = sol.build_radix_tree([list(s) for s in sequences])
            got_norm = sorted((tuple(p), tuple(e)) for p, e in got)
        except Exception:
            ok = 0.0
            break
        if got_norm != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
