import random

def generate_graphs():
    random.seed(42)
    graphs = []
    op_pool = ["Conv2D", "DepthwiseConv2D", "Reshape", "CustomOp", "Add", "Mul", "Softmax", "Pad"]
    supported = {"Conv2D", "DepthwiseConv2D", "Add", "Mul", "Reshape"}
    for i in range(12):
        length = 15 + i * 2
        ops = [random.choice(op_pool) for _ in range(length)]
        graphs.append({"ops": ops, "supported": list(supported)})
    return graphs

def count_partitions(graph):
    ops = graph["ops"]
    supported = set(graph["supported"])
    partitions = 0
    in_delegate = False
    for op in ops:
        is_sup = op in supported
        if is_sup and not in_delegate:
            partitions += 1
            in_delegate = True
        elif not is_sup:
            in_delegate = False
    return partitions

def reduce_partitions(graph):
    ops = list(graph["ops"])
    supported = set(graph["supported"])
    substitutions = {"CustomOp": "Add", "Softmax": "Mul"}
    new_ops = [substitutions.get(op, op) for op in ops]
    return {"ops": new_ops, "supported": graph["supported"]}

def latency_curve(graph, xnnpack_base=12.0, gpu_base=6.0):
    parts = count_partitions(graph)
    n = len(graph["ops"])
    xnn = [xnnpack_base + i * 0.25 for i in range(n)]
    gpu = [gpu_base + parts * 1.8 + i * 0.1 for i in range(n)]
    return {"xnnpack": xnn, "gpu": gpu}
