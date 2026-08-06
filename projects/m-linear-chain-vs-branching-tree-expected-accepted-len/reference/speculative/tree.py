def verify_tree_sample(parents, accepts):
    n = len(parents)
    accepted_set = {i for i, acc in enumerate(accepts) if acc}

    if 0 not in accepted_set:
        return []

    valid_nodes = set()
    for i in range(n):
        if i in accepted_set:
            curr = i
            path = []
            possible = True
            while curr != -1:
                if curr not in accepted_set:
                    possible = False
                    break
                path.append(curr)
                curr = parents[curr]
            if possible:
                valid_nodes.add(i)

    children = [[] for _ in range(n)]
    for i in range(1, n):
        if i in valid_nodes:
            p = parents[i]
            if p in valid_nodes:
                children[p].append(i)

    best_path = []

    def dfs(u, current_path):
        nonlocal best_path
        current_path.append(u)
        if len(current_path) > len(best_path):
            best_path = list(current_path)
        for v in children[u]:
            dfs(v, current_path)
        current_path.pop()

    dfs(0, [])
    return best_path
