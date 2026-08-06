IR_CASES = [
    {"id": 1, "ops": 2, "tensor_elements": 64, "expected": "where"},
    {"id": 2, "ops": 1000, "tensor_elements": 64, "expected": "cond"},
    {"id": 3, "ops": 1, "tensor_elements": 1048576, "expected": "where"},
    {"id": 4, "ops": 500, "tensor_elements": 1048576, "expected": "cond"},
    {"id": 5, "ops": 5, "tensor_elements": 1024, "expected": "where"},
    {"id": 6, "ops": 200, "tensor_elements": 1024, "expected": "cond"}
]
