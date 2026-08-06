def extract_blob(blob: bytes):
    """
    Parse the delegate blob.
    Blob format: b"backend:<name>;ops:<count>;flops:<total>"
    Returns: (backend_name: str, num_ops: int, flops: float)
    """
    text = blob.decode("utf-8")
    parts = dict(kv.split(":") for kv in text.split(";"))
    return parts["backend"], int(parts["ops"]), float(parts["flops"])


def measure_delegation(partitioned_ops: list[dict]) -> float:
    """
    Returns the fraction of FLOPs that were delegated.
    """
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
