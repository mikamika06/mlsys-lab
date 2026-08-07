import random

PROFILES = [
    {
        "nodes": [
            {"name": "Node_0", "op_type": "MatMul", "dur": 150.0, "ep": "CUDAExecutionProvider"},
            {"name": "Node_1", "op_type": "Add", "dur": 20.0, "ep": "CUDAExecutionProvider"},
            {"name": "Node_2", "op_type": "MemcpyFromHost", "dur": 10.0, "ep": "CUDAExecutionProvider"},
            {"name": "Node_3", "op_type": "MatMul", "dur": 200.0, "ep": "CUDAExecutionProvider"},
        ],
        "session_duration": 450.0
    },
    {
        "nodes": [
            {"name": "Node_0", "op_type": "Conv", "dur": 500.0, "ep": "CUDAExecutionProvider"},
            {"name": "Node_1", "op_type": "Relu", "dur": 15.0, "ep": "CUDAExecutionProvider"},
            {"name": "Node_2", "op_type": "Memcpy", "dur": 25.0, "ep": "CPUExecutionProvider"},
            {"name": "Node_3", "op_type": "Conv", "dur": 450.0, "ep": "CUDAExecutionProvider"},
            {"name": "Node_4", "op_type": "MemcpyToHost", "dur": 10.0, "ep": "CPUExecutionProvider"},
        ],
        "session_duration": 1050.0
    }
]

def aggregate_op_types(profile):
    res = {}
    for n in profile["nodes"]:
        op = n["op_type"]
        res[op] = res.get(op, 0.0) + n["dur"]
    return res

def locate_boundary_memcpys(profile):
    res = []
    for n in profile["nodes"]:
        op = n["op_type"]
        if "Memcpy" in op:
            res.append(n)
    return res

def compute_overhead(profile):
    total_nodes = sum(n["dur"] for n in profile["nodes"])
    return profile["session_duration"] - total_nodes
