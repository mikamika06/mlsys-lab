import numpy as np


def mro_isinstance(mro_adj: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    n = mro_adj.shape[0]
    reach = np.zeros((n, n), dtype=bool)
    for start in range(n):
        stack = [start]
        while stack:
            node = stack.pop()
            if reach[start, node]:
                continue
            reach[start, node] = True
            for nxt in range(n):
                if mro_adj[node, nxt] and not reach[start, nxt]:
                    stack.append(nxt)
    return np.array([reach[int(a), int(b)] for a, b in pairs], dtype=bool)
