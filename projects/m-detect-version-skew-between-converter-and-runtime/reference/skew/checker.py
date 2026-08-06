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
