import numpy as np

def get_minimal_blocklist(model_type):
    if model_type == "llama":
        return ["LayerNorm", "Softmax", "Add"]
    return ["LayerNorm", "Softmax"]

def optimize_graph(graph, mode):
    if mode == "transformers":
        return graph + "_opt_trans"
    elif mode == "all":
        return graph + "_opt_all"
    return graph

def evaluate_error(optimized_graph, baseline_outputs):
    if "all" in optimized_graph:
        return 0.05
    return 0.0001

CONFIGS = ["llama", "gpt2"]
