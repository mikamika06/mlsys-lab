import numpy as np

class KDNode:
    __slots__ = ("idx", "split_dim", "left", "right")
    def __init__(self, idx, split_dim):
        self.idx = idx          # index of the point stored at this node
        self.split_dim = split_dim  # dimension used for splitting
        self.left = None
        self.right = None

class KDTree:
    __slots__ = ("root", "points")
    def __init__(self, root, points):
        self.root = root
        self.points = points

def build_kd_tree(points: np.ndarray) -> KDTree:
    """Build a balanced kd‑tree from points."""
    def _build(indices, depth):
        if len(indices) == 0:
            return None
        k = points.shape[1]
        axis = depth % k
        # sort indices by the chosen axis and pick median
        sorted_idx = indices[np.argsort(points[indices, axis])]
        median = len(sorted_idx) // 2
        node = KDNode(sorted_idx[median], axis)
        node.left = _build(sorted_idx[:median], depth + 1)
        node.right = _build(sorted_idx[median+1:], depth + 1)
        return node

    all_indices = np.arange(points.shape[0])
    root = _build(all_indices, 0)
    return KDTree(root, points)

def query_kd_tree(tree: KDTree, point: np.ndarray) -> int:
    """Return the index of the nearest neighbour to `point`."""
    best_idx = None
    best_dist_sq = float("inf")

    def _search(node):
        nonlocal best_idx, best_dist_sq
        if node is None:
            return
        pt = tree.points[node.idx]
        dist_sq = np.sum((pt - point) ** 2)
        if (dist_sq < best_dist_sq or
                (dist_sq == best_dist_sq and node.idx < best_idx)):
            best_dist_sq = dist_sq
            best_idx = node.idx

        axis = node.split_dim
        diff = point[axis] - pt[axis]
        first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)

        _search(first)
        if diff ** 2 < best_dist_sq:
            _search(second)

    _search(tree.root)
    return int(best_idx)
