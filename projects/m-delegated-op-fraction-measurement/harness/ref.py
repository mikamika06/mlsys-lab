import numpy as np

def generate_model(seed, n):
    rng = np.random.RandomState(seed)
    opcodes = [
        "CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MUL",
        "FULLY_CONNECTED", "SOFTMAX", "RESHAPE",
        "CUSTOM", "RSQRT", "GELU"
    ]
    ops = []
    for i in range(n):
        ops.append({
            "id": i,
            "opcode": rng.choice(opcodes),
            "flops": float(rng.randint(100, 10000))
        })
    return ops

MODELS = [generate_model(s, 20 + s * 10) for s in range(5)]

def extract_blob(blob: bytes):
    text = blob.decode("utf-8")
    parts = dict(kv.split(":") for kv in text.split(";"))
    return parts["backend"], int(parts["ops"]), float(parts["flops"])

def measure_delegation(partitioned_ops: list[dict]) -> float:
    total_flops = 0.0
    delegated_flops = 0.0
    for op in partitioned_ops:
        if op["opcode"] == "DELEGATE":
            _, _, flops = extract_blob(op["blob"])
            delegated_flops += flops
            total_flops += flops
        else:
            total_flops += op["flops"]
    return delegated_flops / total_flops if total_flops > 0 else 0.0

def _partition(ops, supported_set, backend_name):
    out = []
    group_len = 0
    group_flops = 0.0
    for op in ops:
        if op["opcode"] in supported_set:
            group_len += 1
            group_flops += op["flops"]
        else:
            if group_len > 0:
                blob = f"backend:{backend_name};ops:{group_len};flops:{group_flops}".encode("utf-8")
                out.append({"opcode": "DELEGATE", "blob": blob})
                group_len = 0
                group_flops = 0.0
            out.append(op)
    if group_len > 0:
        blob = f"backend:{backend_name};ops:{group_len};flops:{group_flops}".encode("utf-8")
        out.append({"opcode": "DELEGATE", "blob": blob})
    return out

def partition_xnnpack(ops):
    return _partition(ops, {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MUL"}, "XNNPACK")

def partition_coreml(ops):
    return _partition(ops, {"CONV_2D", "FULLY_CONNECTED", "ADD", "SOFTMAX", "RESHAPE"}, "CoreML")
