import torch


def safe_convert_or_diagnose(model, example_inputs, unsupported_op_names):
    model.eval()
    try:
        traced = torch.jit.trace(model, example_inputs)
        graph_str = str(traced.graph)

        found = [op for op in unsupported_op_names if op in graph_str]
        if found:
            return {
                "success": False,
                "error_msg": f"Unsupported operators detected in graph: {found}",
                "unsupported_found": found,
            }

        return {
            "success": True,
            "error_msg": "",
            "unsupported_found": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error_msg": str(e),
            "unsupported_found": list(unsupported_op_names),
        }
