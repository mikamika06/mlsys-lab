"""Draft export triage module."""


def run_draft_export(model_spec):
    name = model_spec.get("name", "unknown")
    nodes = model_spec.get("nodes", [])
    has_dynamic = model_spec.get("dynamic_shapes", False)

    issues = []

    for node in nodes:
        op = node.get("op")
        if op == "python_side_effect":
            issues.append({
                "node": node.get("name"),
                "code": "SIDE_EFFECT",
                "message": "Unsupported side effect during tracing"
            })
        elif op == "data_dependent_control_flow":
            issues.append({
                "node": node.get("name"),
                "code": "GUARD_VIOLATION",
                "message": "Data dependent control flow without dynamic guard"
            })
        elif op == "unsupported_aten_op":
            issues.append({
                "node": node.get("name"),
                "code": "UNSUPPORTED_OP",
                "message": "Operator missing ATen decomposition"
            })
        elif op == "dynamic_shape_mismatch" and not has_dynamic:
            issues.append({
                "node": node.get("name"),
                "code": "GUARD_VIOLATION",
                "message": "Shape mismatch without dynamic dimension declaration"
            })

    success = len(issues) == 0
    return {
        "model_name": name,
        "success": success,
        "issues": issues,
        "node_count": len(nodes)
    }
