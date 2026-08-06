def analyze_fallback_causes(verbose_logs):
    fallbacks = []
    for line in verbose_logs:
        line = line.strip()
        if not line.startswith("onednn,exec") and not line.startswith("onednn,create"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        
        engine_info = parts[1] if len(parts) > 1 else ""
        prim_kind = parts[2] if len(parts) > 2 else ""
        impl = parts[3] if len(parts) > 3 else ""
        prop = parts[4] if len(parts) > 4 else ""
        shape_info = parts[5] if len(parts) > 5 else ""
        aux = ",".join(parts[6:]) if len(parts) > 6 else ""

        if "ref" in impl.lower() or "reference" in impl.lower():
            reason = "unsupported_isa"
            if "unaligned" in line.lower() or "nchw" in shape_info.lower() and "nhwc" in line.lower():
                reason = "unaligned_layout"
            elif "f32" in shape_info.lower() and ("amx" in line.lower() or "int8" in line.lower()):
                reason = "unsupported_datatype"
            elif "layout" in aux.lower() or "strides" in aux.lower():
                reason = "unaligned_layout"
            elif "dt" in aux.lower() or "data_type" in aux.lower():
                reason = "unsupported_datatype"

            fallbacks.append({
                "primitive": prim_kind,
                "implementation": impl,
                "reason": reason,
                "shape": shape_info
            })
    return fallbacks
