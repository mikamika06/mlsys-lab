import torch

def count_graph_breaks(fn, args):
    explanation = torch._dynamo.explain(fn)(*args)
    return explanation.graph_break_count

def rank_snippets(snippets_dict, args):
    results = []
    for name, fn in snippets_dict.items():
        breaks = count_graph_breaks(fn, args)
        results.append((breaks, name))
    results.sort(key=lambda x: (x[0], x[1]))
    return [name for _, name in results]

def predict_nested_if_graphs():
    return 3
