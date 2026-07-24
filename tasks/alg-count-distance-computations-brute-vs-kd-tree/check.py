import numpy as np

class _KDNode:
    __slots__ = ("point", "idx", "axis", "left", "right")
    def __init__(self, point, idx, axis):
        self.point = point
        self.idx = idx
        self.axis = axis
        self.left = None
        self.right = None

def _build(points, indices, depth=0):
    if len(indices) == 0:
        return None
    axis = depth % points.shape[1]
    sorted_idx = indices[np.argsort(points[indices, axis])]
    median = len(sorted_idx)//2
    node = _KDNode(points[sorted_idx[median]], sorted_idx[median], axis)
    node.left = _build(points, sorted_idx[:median], depth+1)
    node.right = _build(points, sorted_idx[median+1:], depth+1)
    return node

def _search(node, q, query_idx, k, best, counter):
    if node is None:
        return
    diff = q[node.axis] - node.point[node.axis]
    first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)
    # Evaluate distance only when not the query point itself
    if node.idx != query_idx:
        dist2 = np.sum((q - node.point)**2)
        counter[0] += 1
        if len(best) < k:
            best.append(dist2)
            best.sort()
        else:
            if dist2 < best[-1]:
                best[-1] = dist2
                best.sort()
    _search(first, q, query_idx, k, best, counter)
    if len(best) < k or diff**2 < best[-1]:
        _search(second, q, query_idx, k, best, counter)

def _kd_tree_counts(points, k):
    n = points.shape[0]
    root = _build(points, np.arange(n))
    counter = [0]  # mutable counter
    for i in range(n):
        best = []
        _search(root, points[i], i, k, best, counter)
    return counter[0]

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (rng.standard_normal((4, 2)), 1),
        (rng.standard_normal((10, 5)), 3),
        (rng.standard_normal((20, 3)), 5),
    ]
    ok = 1.0
    for points, k in cases:
        try:
            got = sol.count_distance_computations(points, k)
            if not isinstance(got, tuple) or len(got) != 2:
                ok = 0.0
                break
            brute_ref = points.shape[0] * (points.shape[0]-1)
            kd_ref = _kd_tree_counts(points, k)
            if got[0] != brute_ref or got[1] != kd_ref:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
