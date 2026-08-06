import sys

sys.path.insert(0, ".")
from dynshape.count import count_graphs


def test_graph_count_bounded():
    def f(x):
        return x + 1

    shapes = [(2, 16), (4, 16), (8, 16)]
    cnt = count_graphs(f, shapes)
    assert cnt <= len(shapes), f"graph count {cnt} exceeded shape variation count"
    assert cnt > 0, "graph count cannot be zero"
