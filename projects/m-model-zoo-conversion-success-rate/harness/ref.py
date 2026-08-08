import struct

def make_binary_sigdef(inputs, outputs):
    buf = bytearray()
    buf.extend(struct.pack(">HH", len(inputs), len(outputs)))
    for item in inputs:
        name_b = item["name"].encode("utf-8")
        buf.extend(struct.pack(">H", len(name_b)))
        buf.extend(name_b)
        dtype_b = item["dtype"].encode("utf-8")
        buf.extend(struct.pack(">H", len(dtype_b)))
        buf.extend(dtype_b)
        shape = item["shape"]
        buf.extend(struct.pack(">B", len(shape)))
        for dim in shape:
            buf.extend(struct.pack(">I", dim))
    for item in outputs:
        name_b = item["name"].encode("utf-8")
        buf.extend(struct.pack(">H", len(name_b)))
        buf.extend(name_b)
        dtype_b = item["dtype"].encode("utf-8")
        buf.extend(struct.pack(">H", len(dtype_b)))
        buf.extend(dtype_b)
        shape = item["shape"]
        buf.extend(struct.pack(">B", len(shape)))
        for dim in shape:
            buf.extend(struct.pack(">I", dim))
    return bytes(buf)

SIGNATURE_TESTS = [
    {
        "inputs": [{"name": "input_tensor", "dtype": "float32", "shape": [1, 3, 224, 224]}],
        "outputs": [{"name": "output_tensor", "dtype": "float32", "shape": [1, 1000]}]
    },
    {
        "inputs": [
            {"name": "ids", "dtype": "int32", "shape": [1, 128]},
            {"name": "mask", "dtype": "int32", "shape": [1, 128]}
        ],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 128, 32000]}]
    },
    {
        "inputs": [],
        "outputs": [{"name": "scalar_out", "dtype": "float32", "shape": []}]
    }
]

ERROR_LOG_TESTS = [
    ("RuntimeError: OpCode not found: CustomAttention", "unsupported_op"),
    ("ValueError: shape mismatch: input [1, 64] vs weight [32, 32]", "shape_mismatch"),
    ("Quantization error: calibration failed for tensor x", "quantization_error"),
    ("Out of memory: allocation overflow in arena", "memory_limit"),
    ("Unknown internal compiler error", "unknown")
]

SUCCESS_RECORDS = [
    [{"status": "success"}, {"status": "success"}],
    [{"status": "success"}, {"status": "failed"}, {"status": "success"}],
    [],
    [{"status": "failed"}]
]
