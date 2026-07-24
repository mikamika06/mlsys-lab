def optimize_lru_cache(ops, budget):
    class Node:
        def __init__(self):
            self.children = {}
            self.key = None

    def insert_radix(root, key):
        bits = bin(key)[2:].zfill(16)
        node = root
        for bit in bits:
            if bit not in node.children:
                node.children[bit] = Node()
            node = node.children[bit]
        node.key = key

    def contains(root, key):
        bits = bin(key)[2:].zfill(16)
        node = root
        for bit in bits:
            if bit not in node.children:
                return False
            node = node.children[bit]
        return node.key == key

    root = Node()
    recency = {}
    tick = 0
    hits = 0

    for op, key in ops:
        if op == "query":
            if contains(root, key) and key in recency:
                hits += 1
                tick += 1
                recency[key] = tick
        else:
            if key not in recency:
                insert_radix(root, key)
            tick += 1
            recency[key] = tick
            while len(recency) > budget:
                victim = min(recency, key=lambda k: recency[k])
                del recency[victim]

    return sorted(recency.keys()), hits
