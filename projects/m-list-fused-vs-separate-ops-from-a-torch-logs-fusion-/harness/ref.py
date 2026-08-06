def get_trace_samples():
    return [
        """
        FUSED: add, relu, mul
        SEPARATE: sum, mm
        FUSED: abs, neg
        """,
        """
        SEPARATE: conv2d
        FUSED: clamp, add
        SEPARATE: reshape
        """,
    ]


def get_graph_samples():
    return [
        [
            {"id": "buf0", "op": "add", "inputs": [], "shape": (64, 64), "is_pointwise": True},
            {"id": "buf1", "op": "relu", "inputs": ["buf0"], "shape": (64, 64), "is_pointwise": True},
            {"id": "buf2", "op": "mul", "inputs": ["buf1"], "shape": (64, 64), "is_pointwise": True},
            {"id": "buf3", "op": "sum", "inputs": ["buf2"], "shape": (64,), "is_pointwise": False},
        ],
        [
            {"id": "n0", "op": "exp", "inputs": [], "shape": (32, 32), "is_pointwise": True},
            {"id": "n1", "op": "abs", "inputs": ["n0"], "shape": (32, 32), "is_pointwise": True},
            {"id": "n2", "op": "neg", "inputs": ["n0"], "shape": (32, 32), "is_pointwise": True},
            {"id": "n3", "op": "add", "inputs": ["n1", "n2"], "shape": (32, 32), "is_pointwise": True},
        ]
    ]
