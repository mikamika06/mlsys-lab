import torch


def extract_graph_breaks(model, sample_inputs):
    if not isinstance(sample_inputs, (list, tuple)):
        sample_inputs = (sample_inputs,)
    explanation = torch._dynamo.explain(model)(*sample_inputs)
    reasons = []
    if hasattr(explanation, "graph_break_reasons") and explanation.graph_break_reasons:
        for r in explanation.graph_break_reasons:
            reasons.append(str(r))
    elif hasattr(explanation, "break_reasons") and explanation.break_reasons:
        for r in explanation.break_reasons:
            reasons.append(str(r))
    else:
        reasons = [str(b) for b in getattr(explanation, "graph_breaks", [])]
    return {
        "count": len(reasons),
        "reasons": reasons
    }
