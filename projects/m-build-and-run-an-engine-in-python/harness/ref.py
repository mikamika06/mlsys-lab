SUPPORTED_OPS = {"Relu", "Gemm", "Identity"}

MODELS = [
    {"nodes": [{"op": "Relu"}, {"op": "Gemm"}]},
    {"nodes": [{"op": "Identity"}, {"op": "UnsupportedOp"}]},
]

TRTEXEC_ARGS = [
    {"fp16": True, "batch": 8, "memPoolSize": 2147483648},
    {"int8": True, "batch": 4, "memPoolSize": 1073741824},
]
