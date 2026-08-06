def parse_metadata(header):
    return {
        "magic": header.get("magic"),
        "version": header.get("version"),
        "alignment": header.get("alignment", 32),
        "tensor_count": len(header.get("tensors", [])),
    }
