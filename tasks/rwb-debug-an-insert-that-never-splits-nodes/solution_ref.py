def total_saved_tokens(seqs):
    class Node:
        def __init__(self):
            self.edges = {}

    def insert(node, seq):
        if not seq:
            return

        key = seq[0]
        if key not in node.edges:
            child = Node()
            node.edges[key] = (list(seq), child)
            return

        edge, child = node.edges[key]
        common = 0
        while common < len(edge) and common < len(seq) and edge[common] == seq[common]:
            common += 1

        if common == len(edge):
            insert(child, seq[common:])
            return

        split = Node()
        node.edges[key] = (edge[:common], split)

        old_suffix = edge[common:]
        split.edges[old_suffix[0]] = (old_suffix, child)

        new_suffix = seq[common:]
        if new_suffix:
            new_child = Node()
            split.edges[new_suffix[0]] = (new_suffix, new_child)

    def cost(node):
        total = 0
        for edge, child in node.edges.values():
            total += len(edge)
            total += cost(child)
        return total

    root = Node()
    for seq in seqs:
        insert(root, list(seq))

    return sum(len(seq) for seq in seqs) - cost(root)
