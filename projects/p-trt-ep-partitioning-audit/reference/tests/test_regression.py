import sys

sys.path.insert(0, ".")

from trtep.audit import Graph, Node, partition_graph
from trtep.cache import EngineCache
from trtep.rewriter import rewrite_graph

DEFAULT_SUPPORTED_OPS = {"Conv", "Relu", "Add", "Mul", "MatMul", "Gelu", "Pad", "LayerNormalization", "Softmax"}


def _create_sample_graph():
    nodes = [
        Node("n0", "Conv", ["in0"], ["t0"]),
        Node("n1", "CustomGelu", ["t0"], ["t1"]),
        Node("n2", "Relu", ["t1"], ["out0"])
    ]
    return Graph(nodes, ["in0"], ["out0"])


def test_engine_cache_hits_on_second_lookup():
    cache = EngineCache()
    g = _create_sample_graph()
    subs = partition_graph(g, DEFAULT_SUPPORTED_OPS)
    sub = subs[0]

    build_count = 0

    def builder(s):
        nonlocal build_count
        build_count += 1
        return "engine_blob"

    res1, hit1 = cache.build_or_load(sub, builder)
    assert not hit1
    assert build_count == 1

    res2, hit2 = cache.build_or_load(sub, builder)
    assert hit2
    assert build_count == 1


def test_rewriter_reduces_subgraph_breaks():
    g = _create_sample_graph()
    before_subs = partition_graph(g, DEFAULT_SUPPORTED_OPS)
    assert len(before_subs) == 3

    rewritten = rewrite_graph(g, DEFAULT_SUPPORTED_OPS)
    after_subs = partition_graph(rewritten, DEFAULT_SUPPORTED_OPS)
    assert len(after_subs) < len(before_subs)
    assert len(after_subs) == 1
