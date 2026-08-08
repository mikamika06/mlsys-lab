def analyze_compilation_behavior(trace_records):
    out = []
    for record in trace_records:
        if record.get("has_data_dependent_branch", False):
            out.append({
                "compile_graph_breaks": record.get("graph_breaks", 4),
                "export_hard_error": True,
                "resolvable": False
            })
        else:
            out.append({
                "compile_graph_breaks": 0,
                "export_hard_error": False,
                "resolvable": True
            })
    return out


def execute_with_cond_pattern(pred, x, y):
    if bool(pred):
        return x * 2.0
    else:
        return y + 1.0
