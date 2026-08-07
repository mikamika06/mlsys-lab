from trtep.audit import Graph, Node, find_partition_breakers, partition_graph
from trtep.cache import EngineCache
from trtep.rewriter import rewrite_graph

DEFAULT_SUPPORTED_OPS = {"Conv", "Relu", "Add", "Mul", "MatMul", "Gelu", "Pad", "LayerNormalization", "Softmax"}


def build_benchmark_graph():
    nodes = [
        Node("n0", "Conv", ["in0"], ["t0"]),
        Node("n1", "Relu", ["t0"], ["t1"]),
        Node("n2", "Add", ["t1", "in1"], ["t2"]),
        Node("n3", "Mul", ["t2"], ["t3"]),
        Node("n4", "CustomGelu", ["t3"], ["t4"]),
        Node("n5", "MatMul", ["t4"], ["t5"]),
        Node("n6", "Add", ["t5"], ["t6"]),
        Node("n7", "Relu", ["t6"], ["t7"]),
        Node("n8", "UnsupportedPad", ["t7"], ["t8"]),
        Node("n9", "Conv", ["t8"], ["t9"]),
        Node("n10", "Add", ["t9"], ["t10"]),
        Node("n11", "Relu", ["t10"], ["t11"]),
        Node("n12", "MatMul", ["t11"], ["t12"]),
        Node("n13", "Add", ["t12"], ["t13"]),
        Node("n14", "LegacyNorm", ["t13"], ["t14"]),
        Node("n15", "Softmax", ["t14"], ["out0"])
    ]
    return Graph(nodes, ["in0", "in1"], ["out0"])


ref_partition_graph = partition_graph
ref_find_breakers = find_partition_breakers
ref_rewrite_graph = rewrite_graph
RefEngineCache = EngineCache


def calculate_coverage(graph, supported_ops):
    trt_count = sum(1 for n in graph.nodes if n.op_type in supported_ops)
    return trt_count / len(graph.nodes) if graph.nodes else 0.0
