CASES_M1 = [
    {"op": "MatMul", "has_scale": True, "axis": 0, "dtype": "int8", "expected": "fusion"},
    {"op": "MatMul", "has_scale": True, "axis": 1, "dtype": "int8", "expected": "reformat"},
    {"op": "Gemm", "has_scale": False, "axis": 0, "dtype": "int8", "expected": "reformat"},
    {"op": "MatMul", "has_scale": True, "axis": 0, "dtype": "fp8", "expected": "fusion"},
    {"op": "Add", "has_scale": True, "axis": 0, "dtype": "int8", "expected": "reformat"},
]

CASES_M2 = [
    {"name": "layer1.weight", "shape": (512, 256), "axis": 0},
    {"name": "layer2.weight", "shape": (1024, 512), "axis": 1},
]

def insert_qdq(tensor_info):
    name = tensor_info["name"]
    shape = tensor_info["shape"]
    axis = tensor_info["axis"]
    scale_shape = (shape[axis],) if axis < len(shape) else (1,)
    return {
        "q_node": f"Quantize_{name}",
        "dq_node": f"Dequantize_{name}",
        "scale_shape": scale_shape,
        "axis": axis
    }
