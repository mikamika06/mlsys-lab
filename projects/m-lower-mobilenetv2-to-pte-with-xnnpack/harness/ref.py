CONFIGS = [
    {
        "model": "mobilenet_v2",
        "nodes": [
            {"name": "conv1", "op": "conv2d"},
            {"name": "bn1", "op": "batch_norm"},
            {"name": "relu1", "op": "relu6"},
            {"name": "block1_dw", "op": "conv2d"},
            {"name": "block1_pw", "op": "conv2d"},
            {"name": "add1", "op": "add"},
            {"name": "custom_op", "op": "unsupported_custom", "type": "fp64"}
        ]
    }
]
