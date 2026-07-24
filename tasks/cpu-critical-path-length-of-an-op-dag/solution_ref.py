def critical_path_length(nodes, edges):
    children = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}

    for src, dst in edges:
        children[src].append(dst)
        indegree[dst] += 1

    queue = [node for node in nodes if indegree[node] == 0]
    finish = {node: 0 for node in nodes}

    while queue:
        node = queue.pop(0)
        finish[node] += nodes[node]

        for child in children[node]:
            finish[child] = max(finish[child], finish[node])
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return max(finish.values(), default=0)
