import numpy as np

def get_block_placement(model):
    placement = {}
    for i, block in enumerate(model.get("blocks", [])):
        placement[f"block_{i}"] = block.get("device", "GPU")
    return placement

def find_fallback_ops(model):
    fallbacks = []
    for i, block in enumerate(model.get("blocks", [])):
        if block.get("device") != "ANE":
            fallbacks.append(block.get("fallback_op", f"op_{i}"))
    return fallbacks
