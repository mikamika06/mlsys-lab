def resolve_availability(model_spec, target_version):
    min_target = model_spec.get("minimum_deployment_target", "iOS15")
    ops = model_spec.get("operators", [])

    unsupported = [op for op in ops if op.get("min_version", "iOS15") > target_version]

    if unsupported:
        resolved_ops = [
            {**op, "min_version": target_version} for op in ops
        ]
        return {
            "status": "resolved",
            "adjusted_operators": resolved_ops,
            "success": True
        }

    return {
        "status": "compatible",
        "adjusted_operators": ops,
        "success": True
    }
