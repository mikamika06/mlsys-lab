def compute_vram(model_spec, ngl):
    overhead = model_spec["overhead_bytes"]
    layer_sizes = model_spec["layer_bytes"]
    offloaded = sum(layer_sizes[:ngl])
    ctx = model_spec["ctx_bytes"] if ngl > 0 else 0
    return overhead + offloaded + ctx
