import torch


def save_and_verify_adapter(model, adapter_weights, input_tensor):
    with torch.no_grad():
        out_original = model(input_tensor, adapter_weights)
    saved_state = {k: v.clone() for k, v in adapter_weights.items()}
    with torch.no_grad():
        out_reloaded = model(input_tensor, saved_state)
    return torch.allclose(out_original, out_reloaded, atol=1e-6, rtol=1e-6)
