def _oracle_saved_tokens(seqs):
    class Node:
        def __init__(self):
            self.edges = {}

    def insert(node, seq):
        if not seq:
            return
        first = seq[0]
        if first not in node.edges:
            child = Node()
            node.edges[first] = ([*seq], child)
            return

        edge, child = node.edges[first]
        common = 0
        while common < len(edge) and common < len(seq) and edge[common] == seq[common]:
            common += 1

        if common == len(edge):
            insert(child, seq[common:])
            return

        split = Node()
        node.edges[first] = (edge[:common], split)

        old_suffix = edge[common:]
        old_child = child
        split.edges[old_suffix[0]] = (old_suffix, old_child)

        new_suffix = seq[common:]
        if new_suffix:
            new_child = Node()
            split.edges[new_suffix[0]] = (new_suffix, new_child)

    def edge_cost(node):
        total = 0
        for edge, child in node.edges.values():
            total += len(edge)
            total += edge_cost(child)
        return total

    root = Node()
    for seq in seqs:
        insert(root, list(seq))

    plain = sum(len(seq) for seq in seqs)
    return plain - edge_cost(root)


def grade(sol, fx) -> dict:
    cases = [
        [[1, 2, 3, 4], [1, 2, 5]],
        [[10, 20, 30], [10, 20, 40], [10, 20, 30, 50]],
        [[7, 8, 9], [7, 8, 6], [7, 8, 5], [1, 2]],
        [[3, 4, 5, 6, 7], [3, 4, 8], [3, 9]],
        [[1], [1, 2], [1, 3], [2, 1]],
    ]
    ok = 1.0
    for seqs in cases:
        try:
            got = sol.total_saved_tokens([list(x) for x in seqs])
        except Exception:
            ok = 0.0
            break
        expected = _oracle_saved_tokens([list(x) for x in seqs])
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
