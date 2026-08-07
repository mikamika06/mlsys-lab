def validate_dynamic_conversion(input_shapes: dict[str, tuple], shape_hints: dict[str, dict] | None) -> dict:
    hints = shape_hints or {}
    for name, shape in input_shapes.items():
        dynamic_dims = [i for i, dim in enumerate(shape) if dim == -1 or dim is None]
        if dynamic_dims:
            if name not in hints:
                raise ValueError(f"Dynamic input '{name}' missing shape hints")
            hint_info = hints[name]
            for d in dynamic_dims:
                if d not in hint_info:
                    raise ValueError(f"Input '{name}' dimension {d} missing min/max bounds")
                bounds = hint_info[d]
                if "min" not in bounds or "max" not in bounds or bounds["min"] <= 0 or bounds["max"] < bounds["min"]:
                    raise ValueError(f"Input '{name}' dimension {d} has invalid bounds {bounds}")
    return {"status": "VALID", "inputs_checked": len(input_shapes)}
