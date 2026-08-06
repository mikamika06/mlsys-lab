import struct

def generate_mock_flatbuffer(tensor_configs):
    data = bytearray(b"TFL3")
    for cfg in tensor_configs:
        name = cfg.get("name", "tensor").encode("utf-8")
        scale = cfg.get("scale", 1.0)
        zp = cfg.get("zero_point", 0)
        data.extend(struct.pack("<I", len(name)))
        data.extend(name)
        data.extend(struct.pack("<fI", scale, zp))
    return bytes(data)

def get_test_cases():
    cases = [
        {
            "tensors": [
                {"name": "input_tensor", "scale": 0.1, "zero_point": 0},
                {"name": "weights_degenerate", "scale": 0.0, "zero_point": 128},
                {"name": "bias_ok", "scale": 0.05, "zero_point": 0},
            ],
            "expected_degenerate": [1]
        },
        {
            "tensors": [
                {"name": "dense_w", "scale": 0.02, "zero_point": 0},
                {"name": "dense_b", "scale": 0.04, "zero_point": 0},
            ],
            "expected_degenerate": []
        }
    ]
    return cases
