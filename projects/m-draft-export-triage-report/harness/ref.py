"""Reference oracle data and logic."""

TEST_MODELS = [
    {
        "name": "resnet_block",
        "dynamic_shapes": True,
        "nodes": [
            {"name": "conv1", "op": "aten_conv2d"},
            {"name": "bn1", "op": "aten_batch_norm"},
            {"name": "relu", "op": "aten_relu"}
        ]
    },
    {
        "name": "custom_lstm",
        "dynamic_shapes": False,
        "nodes": [
            {"name": "step_check", "op": "data_dependent_control_flow"},
            {"name": "state_update", "op": "python_side_effect"}
        ]
    },
    {
        "name": "custom_attn",
        "dynamic_shapes": False,
        "nodes": [
            {"name": "matmul", "op": "aten_matmul"},
            {"name": "custom_kernel", "op": "unsupported_aten_op"},
            {"name": "resize", "op": "dynamic_shape_mismatch"}
        ]
    }
]


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


def generate_triage_report(triage_results):
    total_models = len(triage_results)
    successful = sum(1 for r in triage_results if r.get("success", False))
    failed = total_models - successful

    code_counts = {
        "GUARD_VIOLATION": 0,
        "SIDE_EFFECT": 0,
        "UNSUPPORTED_OP": 0,
        "UNKNOWN": 0
    }

    for r in triage_results:
        for issue in r.get("issues", []):
            code = issue.get("code", "UNKNOWN")
            if code in code_counts:
                code_counts[code] += 1
            else:
                code_counts["UNKNOWN"] += 1

    total_issues = sum(code_counts.values())

    weights = {
        "GUARD_VIOLATION": 2,
        "SIDE_EFFECT": 5,
        "UNSUPPORTED_OP": 10,
        "UNKNOWN": 8
    }

    priority_score = sum(code_counts[k] * weights[k] for k in weights)

    if priority_score == 0:
        status = "READY"
    elif priority_score < 15:
        status = "NEEDS_ANNOTATION"
    else:
        status = "REQUIRES_REWRITE"

    return {
        "total_models": total_models,
        "successful_exports": successful,
        "failed_exports": failed,
        "total_issues": total_issues,
        "issue_counts": code_counts,
        "priority_score": priority_score,
        "status": status
    }
