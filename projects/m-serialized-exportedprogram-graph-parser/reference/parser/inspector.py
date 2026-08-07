def get_edges(graph_data):
    nodes = graph_data.get("nodes", [])
    edges = []
    name_to_node = {n["name"]: n for n in nodes}
    for node in nodes:
        for inp in node.get("inputs", []):
            if inp in name_to_node:
                edges.append((inp, node["name"]))
    return sorted(edges)


def topological_sort(graph_data):
    nodes = graph_data.get("nodes", [])
    name_to_node = {n["name"]: n for n in nodes}
    in_degree = {n["name"]: 0 for n in nodes}
    adj = {n["name"]: [] for n in nodes}

    for node in nodes:
        for inp in node.get("inputs", []):
            if inp in name_to_node:
                adj[inp].append(node["name"])
                in_degree[node["name"]] += 1

    queue = [n["name"] for n in nodes if in_degree[n["name"]] == 0]
    queue.sort()
    order = []

    while queue:
        curr = queue.pop(0)
        order.append(curr)
        for nxt in adj[curr]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    return order


def get_node_attributes(graph_data, node_name):
    nodes = graph_data.get("nodes", [])
    for node in nodes:
        if node.get("name") == node_name:
            return dict(node.get("attributes", {}))
    return {}
