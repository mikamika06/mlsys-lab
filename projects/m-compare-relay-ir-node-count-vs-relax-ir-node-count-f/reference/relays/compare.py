def count_nodes(ir_expr):
    if isinstance(ir_expr, dict):
        cnt = 1
        for v in ir_expr.values():
            cnt += count_nodes(v)
        return cnt
    elif isinstance(ir_expr, list):
        cnt = 0
        for item in ir_expr:
            cnt += count_nodes(item)
        return cnt
    return 1

def compare_ir_counts(model):
    relay_ir = {
        "type": "RelayModule",
        "functions": {
            "main": {
                "body": {
                    "op": "Function",
                    "params": model["ops"],
                    "constants": list(model["constants"].keys())
                }
            }
        }
    }
    relax_ir = {
        "type": "RelaxModule",
        "blocks": [
            {
                "binding": "SeqExpr",
                "operations": model["ops"],
                "constants": list(model["constants"].keys())
            }
        ]
    }
    relay_count = count_nodes(relay_ir)
    relax_count = count_nodes(relax_ir)
    return {
        "relay_count": relay_count,
        "relax_count": relax_count,
        "ratio": float(relay_count) / float(relax_count if relax_count > 0 else 1)
    }
