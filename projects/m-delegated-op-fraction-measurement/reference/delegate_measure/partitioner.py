XNNPACK_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MUL"}
COREML_OPS = {"CONV_2D", "FULLY_CONNECTED", "ADD", "SOFTMAX", "RESHAPE"}

def _partition(ops: list[dict], supported_set: set, backend_name: str) -> list[dict]:
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

def partition_xnnpack(ops: list[dict]) -> list[dict]:
    """
    Group contiguous XNNPACK-supported ops into DELEGATE ops.
    """
    return _partition(ops, XNNPACK_OPS, "XNNPACK")

def partition_coreml(ops: list[dict]) -> list[dict]:
    """
    Group contiguous CoreML-supported ops into DELEGATE ops.
    """
    return _partition(ops, COREML_OPS, "CoreML")
