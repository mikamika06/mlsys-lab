CONFIGS = [
    {
        "spec": {
            "has_calibration_data": True,
            "max_calibration_sec": 300,
            "target_bpp": 4.0,
            "accuracy_tolerance": 0.01,
            "model_params_b": 7.0,
        },
        "model": {"total_params": 7_000_000_000, "hidden_dim": 4096, "num_layers": 32},
        "libs": {
            "bitsandbytes": {
                "formats": [
                    {"name": "nf4", "base_bits": 4, "group_size": 64, "scale_bits": 16, "zero_bits": 0, "deprecated": False},
                    {"name": "fp4", "base_bits": 4, "group_size": None, "scale_bits": 0, "zero_bits": 0, "deprecated": False},
                ]
            },
            "auto_gptq": {
                "formats": [
                    {"name": "gptq_g128", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                    {"name": "gptq_act_order_v1", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": True},
                ]
            },
            "auto_awq": {
                "formats": [
                    {"name": "awq_g128", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "vllm_quant": {
                "formats": [
                    {"name": "marlin_fp16", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 0, "deprecated": False},
                ]
            },
        },
        "manifest": {
            "selected": [
                {"lib": "auto_gptq", "format": "gptq_act_order_v1"},
                {"lib": "bitsandbytes", "format": "nf4"},
            ]
        },
    },
    {
        "spec": {
            "has_calibration_data": False,
            "max_calibration_sec": 0,
            "target_bpp": 8.0,
            "accuracy_tolerance": 0.05,
            "model_params_b": 13.0,
        },
        "model": {"total_params": 13_000_000_000, "hidden_dim": 5120, "num_layers": 40},
        "libs": {
            "bitsandbytes": {
                "formats": [
                    {"name": "int8_vector", "base_bits": 8, "group_size": None, "scale_bits": 0, "zero_bits": 0, "deprecated": False},
                ]
            },
            "auto_gptq": {
                "formats": [
                    {"name": "gptq_g64", "base_bits": 4, "group_size": 64, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "auto_awq": {
                "formats": [
                    {"name": "awq_g64", "base_bits": 4, "group_size": 64, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "vllm_quant": {
                "formats": [
                    {"name": "squeezellm", "base_bits": 4, "group_size": None, "scale_bits": 0, "zero_bits": 0, "deprecated": True},
                ]
            },
        },
        "manifest": {
            "selected": [
                {"lib": "vllm_quant", "format": "squeezellm"},
            ]
        },
    },
    {
        "spec": {
            "has_calibration_data": True,
            "max_calibration_sec": 30,
            "target_bpp": 4.0,
            "accuracy_tolerance": 0.03,
            "model_params_b": 70.0,
        },
        "model": {"total_params": 70_000_000_000, "hidden_dim": 8192, "num_layers": 80},
        "libs": {
            "bitsandbytes": {
                "formats": [
                    {"name": "nf4", "base_bits": 4, "group_size": 64, "scale_bits": 16, "zero_bits": 0, "deprecated": False},
                ]
            },
            "auto_gptq": {
                "formats": [
                    {"name": "gptq_g128", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "auto_awq": {
                "formats": [
                    {"name": "awq_g128", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "vllm_quant": {
                "formats": [
                    {"name": "compressed_tensors", "base_bits": 8, "group_size": None, "scale_bits": 0, "zero_bits": 0, "deprecated": False},
                ]
            },
        },
        "manifest": {
            "selected": [
                {"lib": "bitsandbytes", "format": "nf4"},
            ]
        },
    },
    {
        "spec": {
            "has_calibration_data": True,
            "max_calibration_sec": 600,
            "target_bpp": 4.0,
            "accuracy_tolerance": 0.05,
            "model_params_b": 3.0,
        },
        "model": {"total_params": 3_000_000_000, "hidden_dim": 2560, "num_layers": 32},
        "libs": {
            "bitsandbytes": {
                "formats": [
                    {"name": "fp4", "base_bits": 4, "group_size": None, "scale_bits": 0, "zero_bits": 0, "deprecated": False},
                ]
            },
            "auto_gptq": {
                "formats": [
                    {"name": "gptq_g128", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "auto_awq": {
                "formats": [
                    {"name": "awq_g128", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 16, "deprecated": False},
                ]
            },
            "vllm_quant": {
                "formats": [
                    {"name": "marlin_fp16", "base_bits": 4, "group_size": 128, "scale_bits": 16, "zero_bits": 0, "deprecated": False},
                ]
            },
        },
        "manifest": {
            "selected": [
                {"lib": "auto_gptq", "format": "gptq_g128"},
            ]
        },
    }
]

REGISTRY = {
    "auto_gptq": {
        "gptq_act_order_v1": "gptq_g128",
    },
    "vllm_quant": {
        "squeezellm": "marlin_fp16",
    },
    "bitsandbytes": {},
    "auto_awq": {},
}


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


def audit_dependencies(manifest, registry):
    selected = manifest.get("selected", [])
    deprecated_count = 0
    warnings = []
    remediations = {}
    for item in selected:
        lib = item["lib"]
        fmt = item["format"]
        dep_map = registry.get(lib, {})
        if fmt in dep_map:
            deprecated_count += 1
            replacement = dep_map[fmt]
            warnings.append(f"{lib}.{fmt} is deprecated")
            remediations[f"{lib}.{fmt}"] = replacement

    valid = (deprecated_count == 0)
    return {
        "valid": valid,
        "deprecated_count": deprecated_count,
        "warnings": warnings,
        "remediations": remediations,
    }
