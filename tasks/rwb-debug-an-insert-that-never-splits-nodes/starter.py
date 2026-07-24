def total_saved_tokens(seqs):
    class Node:
        def __init__(self):
            self.edges = {}

    def insert(node, seq):
        if not seq:
            return
        key = seq[0]
        if key not in node.edges:
            node.edges[key] = (list(seq), Node())
            return

        edge, child = node.edges[key]
        if seq[:len(edge)] == edge:
            insert(child, seq[len(edge):])
        else:
            # BUG: a partial edge match should split the edge, but this code
            # incorrectly creates a sibling at the current node.
            node.edges[seq[0] + 1000000] = (list(seq), Node())

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
