def make_3op_model():
    return {
        "name": "model_3op",
        "ops": [
            {"op": "add", "inputs": ["input_0", "constant_weight"]},
            {"op": "multiply", "inputs": ["previous", "constant_scale"]},
            {"op": "relu", "inputs": ["previous"]}
        ],
        "constants": {
            "constant_weight": [1.0, 2.0, 3.0, 4.0],
            "constant_scale": [0.5, 0.5, 0.5, 0.5]
        }
    }
