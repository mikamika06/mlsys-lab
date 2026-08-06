import torch


def audit_determinism(model, inputs, runs=5):
    outputs = []
    with torch.no_grad():
        for _ in range(runs):
            out = model(*inputs)
            if isinstance(out, (tuple, list)):
                out = tuple(o.clone() for o in out)
            else:
                out = out.clone()
            outputs.append(out)
    first = outputs[0]
    is_deterministic = True
    max_diff = 0.0
    for other in outputs[1:]:
        if isinstance(first, tuple):
            for f_elem, o_elem in zip(first, other):
                diff = torch.max(torch.abs(f_elem - o_elem)).item()
                max_diff = max(max_diff, diff)
                if diff > 0.0:
                    is_deterministic = False
        else:
            diff = torch.max(torch.abs(first - other)).item()
            max_diff = max(max_diff, diff)
            if diff > 0.0:
                is_deterministic = False
    return {"is_deterministic": is_deterministic, "max_diff": max_diff, "runs": runs}


def evaluate_gate(audit_results, tolerance=0.0):
    if not audit_results.get("is_deterministic", False):
        return False
    return audit_results.get("max_diff", float("inf")) <= tolerance
