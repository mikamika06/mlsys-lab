class _Node:
    def __init__(self):
        self.children = {}
        self.key = None


def _radix_insert(root, key):
    bits = bin(key)[2:].zfill(16)
    node = root
    for bit in bits:
        node = node.children.setdefault(bit, _Node())
    node.key = key


def _radix_contains(root, key):
    bits = bin(key)[2:].zfill(16)
    node = root
    for bit in bits:
        if bit not in node.children:
            return False
        node = node.children[bit]
    return node.key == key


def _oracle(ops, budget):
    root = _Node()
    cache = {}
    clock = 0
    hits = 0

    for op, key in ops:
        if op == "query":
            if _radix_contains(root, key) and key in cache:
                hits += 1
                clock += 1
                cache[key] = clock
        elif op == "insert":
            if key not in cache:
                _radix_insert(root, key)
            clock += 1
            cache[key] = clock

            while len(cache) > budget:
                victim = min(cache, key=lambda k: cache[k])
                del cache[victim]

    return sorted(cache.keys()), hits


def grade(sol, fx) -> dict:
    cases = [
        [
            ("insert", 10),
            ("insert", 20),
            ("query", 10),
            ("insert", 30),
        ],
        [
            ("insert", 1),
            ("insert", 2),
            ("insert", 3),
            ("query", 1),
            ("insert", 4),
            ("query", 2),
            ("query", 4),
        ],
        [
            ("insert", 100),
            ("insert", 200),
            ("query", 100),
            ("insert", 300),
            ("insert", 400),
            ("query", 300),
            ("query", 100),
        ],
        [
            ("insert", 7),
            ("insert", 15),
            ("query", 7),
            ("query", 7),
            ("insert", 23),
            ("insert", 31),
            ("query", 15),
        ],
    ]
    budgets = [2, 3, 2, 1]

    for ops, budget in zip(cases, budgets):
        try:
            got = sol.optimize_lru_cache(list(ops), budget)
        except Exception:
            return {"exact_match": 0.0}
        if got != _oracle(list(ops), budget):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
