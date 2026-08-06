HEADERS = [
    {"magic": "GGUF", "version": 2, "alignment": 32, "tensors": [1, 2]},
    {"magic": "GGUF", "version": 5, "alignment": 32, "tensors": [1]},
    {"magic": "GGUF", "version": 1, "alignment": 16, "tensors": [1, 2, 3]},
]

CAPS = {
    "min_version": 1,
    "max_version": 3,
    "supported_alignments": [32, 64],
}

def parse_metadata(header):
    return {
        "magic": header.get("magic"),
        "version": header.get("version"),
        "alignment": header.get("alignment", 32),
        "tensor_count": len(header.get("tensors", [])),
    }

def get_supported_versions():
    return CAPS

def detect_skew(metadata, runtime_caps):
    v = metadata.get("version", 0)
    min_v = runtime_caps.get("min_version", 1)
    max_v = runtime_caps.get("max_version", 3)
    align = metadata.get("alignment", 0)
    supported_aligns = runtime_caps.get("supported_alignments", [])
    if v < min_v:
        return {"status": "incompatible", "reason": "version_too_old"}
    if v > max_v:
        return {"status": "incompatible", "reason": "version_too_new"}
    if align not in supported_aligns:
        return {"status": "incompatible", "reason": "unsupported_alignment"}
    return {"status": "compatible", "reason": "ok"}
