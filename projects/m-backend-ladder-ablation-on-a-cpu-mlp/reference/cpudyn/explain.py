import torch


def inspect_dynamo_explain(model, example_inputs):
    """Inspect model using torch._dynamo.explain and extract graph metrics."""
    explanation = torch._dynamo.explain(model)(*example_inputs)
    
    graph_count = len(explanation.graphs)
    graph_break_count = len(explanation.break_reasons)
    
    break_reasons = []
    for reason in explanation.break_reasons:
        msg = getattr(reason, "reason", str(reason))
        break_reasons.append(msg)
        
    return {
        "graph_count": graph_count,
        "graph_break_count": graph_break_count,
        "break_reasons": break_reasons,
        "ops_per_graph": [len(g.nodes) for g in explanation.graphs]
    }
