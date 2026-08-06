from offload.compute import compute_vram

def max_ngl_for_budget(model_spec, budget_bytes):
    best = 0
    for ngl in range(len(model_spec["layer_bytes"]) + 1):
        if compute_vram(model_spec, ngl) <= budget_bytes:
            best = ngl
    return best
