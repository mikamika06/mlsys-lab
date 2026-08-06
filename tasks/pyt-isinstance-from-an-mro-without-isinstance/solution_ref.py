def mro_isinstance(mro_adj: list[list[int]], pairs: list[list[int]]) -> list[bool]:
    n = len(mro_adj)
    reach = [[False] * n for _ in range(n)]
    for start in range(n):
        stack = [start]
        while stack:
            node = stack.pop()
            if reach[start][node]:
                continue
            reach[start][node] = True
            for nxt in range(n):
                if mro_adj[node][nxt] and not reach[start][nxt]:
                    stack.append(nxt)
    return [reach[int(a)][int(b)] for a, b in pairs]
