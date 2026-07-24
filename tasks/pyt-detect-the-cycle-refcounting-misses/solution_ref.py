def uncollectable_cycles(graph):
    index = 0
    indices = {}
    low = {}
    stack = []
    on_stack = set()
    components = []

    def dfs(v):
        nonlocal index
        indices[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph[v]:
            if w not in indices:
                dfs(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])

        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            components.append(comp)

    for node in graph:
        if node not in indices:
            dfs(node)

    answer = []
    for comp in components:
        members = set(comp)
        if len(members) == 1:
            node = comp[0]
            if node not in graph[node]:
                continue

        has_external = False
        for src in graph:
            if src not in members:
                for dst in graph[src]:
                    if dst in members:
                        has_external = True
                        break
            if has_external:
                break

        if not has_external:
            answer.append(sorted(members))

    return sorted(answer, key=lambda x: x[0])
