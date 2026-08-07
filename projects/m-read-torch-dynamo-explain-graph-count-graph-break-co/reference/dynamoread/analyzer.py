import torch


def analyze_breaks(fn, sample_args):
    explanation = torch._dynamo.explain(fn)(*sample_args)
    return {
        "graph_count": int(explanation.graph_count),
        "graph_break_count": int(explanation.graph_break_count),
    }
