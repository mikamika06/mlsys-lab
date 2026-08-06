import sys


def aggregate_footprint(adjacency: list[list[int]], payloads: list[list[int]]) -> float:
    nodes = []
    for payload in payloads:
        nodes.append({"payload": list(payload), "children": []})

    n = len(nodes)
    for i in range(n):
        for j in range(n):
            if adjacency[i][j] != 0:
                nodes[i]["children"].append(nodes[j])

    seen = set()

    def deep_size(obj):
        oid = id(obj)
        if oid in seen:
            return 0
        seen.add(oid)

        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            for key, value in obj.items():
                total += deep_size(key)
                total += deep_size(value)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                total += deep_size(item)
        return total

    total = 0
    for node in nodes:
        total += deep_size(node)

    flat_count = sum(len(payload) for payload in payloads)
    flat_bytes = flat_count * 8
    return float(total / flat_bytes)
