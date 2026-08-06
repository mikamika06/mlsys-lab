def classify_decision_rule(spec):
    if not spec.get("has_calibration_data", False) or spec.get("max_calibration_sec", 0) < 60:
        strategy = "calibration_free"
        if spec.get("target_bpp", 16.0) >= 8.0:
            method = "int8_weight_only"
        else:
            method = "rtn_4bit"
    else:
        strategy = "calibration_based"
        if spec.get("accuracy_tolerance", 1.0) < 0.02 or spec.get("target_bpp", 16.0) <= 3.5:
            method = "awq"
        else:
            method = "gptq"
    return {
        "strategy": strategy,
        "method": method,
        "requires_dataset": (strategy == "calibration_based"),
    }


def build_bpp_table(model_spec, library_configs):
    total_params = model_spec["total_params"]
    rows = []
    for lib_name in sorted(library_configs.keys()):
        lib_data = library_configs[lib_name]
        for fmt in lib_data.get("formats", []):
            base_bits = float(fmt["base_bits"])
            g = fmt.get("group_size")
            if g and g > 0:
                overhead = (float(fmt.get("scale_bits", 0)) + float(fmt.get("zero_bits", 0))) / float(g)
            else:
                overhead = 0.0
            bpp = round(base_bits + overhead, 4)
            size_mb = round((total_params * bpp) / (8.0 * 1024.0 * 1024.0), 2)
            rows.append({
                "library": lib_name,
                "format": fmt["name"],
                "bpp": bpp,
                "size_mb": size_mb,
                "deprecated": bool(fmt.get("deprecated", False)),
            })
    return sorted(rows, key=lambda x: (x["library"], x["format"]))
