import numpy as np

def parse_verbose_line(line):
    parts = line.strip().split(",")
    if len(parts) < 8 or parts[0] != "onednn_verbose":
        return None
    prim = parts[3]
    jit = parts[4]
    status = parts[5] if len(parts) > 5 else ""
    dims_str = ""
    for p in parts:
        if "mb" in p or "ic" in p:
            dims_str = p
            break
    fallback_reason = "none"
    if "ref" in jit or "fallback" in line:
        fallback_reason = "format_mismatch" if "format" in line else "unsupported_config"
    return {
        "primitive": prim,
        "jit": jit,
        "status": status,
        "dims": dims_str,
        "fallback_reason": fallback_reason
    }
