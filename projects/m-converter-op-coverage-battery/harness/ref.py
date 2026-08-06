import numpy as np

TEST_RUNTIMES = [
    {
        "name": "EdgeAccelerator_V1",
        "supported_ops": ["Conv2D", "Relu", "Add", "Mul"],
        "decomposable_ops": {
            "GELU": ["Mul", "Add", "Erf"],
            "LayerNorm": ["Mean", "Sub", "Square", "Add", "Sqrt", "Div"]
        }
    },
    {
        "name": "MicroDSP_V2",
        "supported_ops": ["Conv2D", "Add"],
        "decomposable_ops": {
            "Relu": ["Max"]
        }
    }
]

TEST_GRAPHS = [
    {
        "nodes": [
            {"id": "n0", "op_type": "Conv2D", "inputs": ["in"]},
            {"id": "n1", "op_type": "GELU", "inputs": ["n0"]},
            {"id": "n2", "op_type": "CustomLayer", "inputs": ["n1"]}
        ]
    },
    {
        "nodes": [
            {"id": "n0", "op_type": "LayerNorm", "inputs": ["in"]},
            {"id": "n1", "op_type": "Add", "inputs": ["n0"]},
            {"id": "n2", "op_type": "UnknownOp", "inputs": ["n1"]}
        ]
    }
]

TEST_METRICS = {
    "GELU": {"custom_op_overhead": 12.5, "fallback_exec": 4.0, "decomposition_exec": 3.2},
    "LayerNorm": {"custom_op_overhead": 5.0, "fallback_exec": 10.0, "decomposition_exec": 18.0},
    "CustomLayer": {"custom_op_overhead": 20.0, "fallback_exec": 15.0, "decomposition_exec": 50.0}
}


def gelu_decompose(node):
    return [
        {"id": f"{node['id']}_mul", "op_type": "Mul", "inputs": node["inputs"]},
        {"id": f"{node['id']}_erf", "op_type": "Erf", "inputs": [f"{node['id']}_mul"]}
    ]


EQUIVALENCE_TABLE = {
    "GELU": {"decompose_fn": gelu_decompose}
}
