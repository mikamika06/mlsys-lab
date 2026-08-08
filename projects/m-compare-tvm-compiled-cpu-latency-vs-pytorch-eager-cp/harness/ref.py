MODELS = [
    {"name": "model_alpha", "torch": [15.2, 14.8, 15.0], "tvm": [5.1, 4.9, 5.0], "unsupported_op": None},
    {"name": "model_beta", "torch": [22.1, 21.9], "tvm": [11.0, 11.2], "unsupported_op": None},
    {"name": "model_gamma", "torch": [10.0, 10.0], "tvm": [2.0, 2.0], "unsupported_op": "relax.op.custom_attention"}
]
