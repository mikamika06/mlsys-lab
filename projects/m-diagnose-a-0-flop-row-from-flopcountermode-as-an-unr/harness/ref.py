def get_m1_tests():
    return [
        ("custom_attention", 0, {"linear", "conv2d"}, "unregistered_custom_op"),
        ("aten::linear", 1000, {"linear", "conv2d"}, "valid"),
        ("custom_norm", 0, {"linear"}, "unregistered_custom_op"),
        ("aten::matmul", 500, {"linear", "matmul"}, "valid"),
    ]

def get_m2_tests():
    return [
        ({"num_layers": 32, "hidden_size": 4096, "intermediate_size": 11008}, 128),
        ({"num_layers": 16, "hidden_size": 2048, "intermediate_size": 5504}, 64),
        ({"num_layers": 24, "hidden_size": 3072, "intermediate_size": 8192}, 96),
    ]
