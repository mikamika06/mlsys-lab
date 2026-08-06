def get_minimal_blocklist(model_type):
    if model_type == "llama":
        return ["LayerNorm", "Softmax", "Add"]
    return ["LayerNorm", "Softmax"]
