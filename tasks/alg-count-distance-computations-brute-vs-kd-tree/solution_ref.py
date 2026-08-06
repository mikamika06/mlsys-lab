import math
import numpy as np


class _KDNode:
    __slots__ = ("point", "idx", "axis", "left", "right")

    def __init__(self, point, idx, axis):
        self.point = point
        self.idx = idx
        self.axis = axis
        self.left = None
        self.right = None


def _sort_indices(indices, points, axis):
    idx_list = [idx for idx in indices]
    n = len(idx_list)
    for i in range(1, n):
        key = idx_list[i]
        key_val = points[key, axis]
        j = i - 1
        while j >= 0 and points[idx_list[j], axis] > key_val:
            idx_list[j + 1] = idx_list[j]
            j -= 1
        idx_list[j + 1] = key
    return idx_list


def _build(points, indices, depth=0):
    if len(indices) == 0:
        return None
    axis = depth % points.shape[1]
    sorted_idx = _sort_indices(indices, points, axis)
    median = len(sorted_idx) // 2
    node = _KDNode(points[sorted_idx[median]], sorted_idx[median], axis)
    node.left = _build(points, sorted_idx[:median], depth + 1)
    node.right = _build(points, sorted_idx[median + 1 :], depth + 1)
    return node


def _sort_best(best):
    n = len(best)
    for i in range(1, n):
        key = best[i]
        j = i - 1
        while j >= 0 and best[j] > key:
            best[j + 1] = best[j]
            j -= 1
        best[j + 1] = key


def _search(node, q, query_idx, k, best, counter):
    if node is None:
        return
    diff = q[node.axis] - node.point[node.axis]
    first, second = (node.left, node.right) if diff <= 0 else (node.right, node.left)
    if node.idx != query_idx:
        dist2 = 0.0
        for d in range(len(q)):
            delta = q[d] - node.point[d]
            dist2 += delta * delta
        counter[0] += 1
        if len(best) < k:
            best.append(dist2)
            _sort_best(best)
        else:
            if dist2 < best[-1]:
                best[-1] = dist2
                _sort_best(best)
    _search(first, q, query_idx, k, best, counter)
    if len(best) < k or diff * diff < best[-1]:
        _search(second, q, query_idx, k, best, counter)


def count_distance_computations(points: np.ndarray, k: int) -> tuple[int, int]:
    """
    Return the exact number of distance evaluations performed by a
    brute‑force search and by a kd‑tree search.
    """
    n = points.shape[0]
    brute_count = n * (n - 1)
    initial_indices = [i for i in range(n)]
    root = _build(points, initial_indices)
    counter = [0]
    for i in range(n):
        best = []
        _search(root, points[i], i, k, best, counter)
    kd_count = counter[0]
    return brute_count, kd_count
