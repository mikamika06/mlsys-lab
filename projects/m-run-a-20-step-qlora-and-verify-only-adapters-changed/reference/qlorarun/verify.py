import torch


def verify_adapters_only_changed(init_weights, final_weights):
    base_changed = False
    adapters_changed = False
    for name in init_weights:
        init_p = init_weights[name]
        fin_p = final_weights[name]
        diff = torch.max(torch.abs(init_p - fin_p)).item()
        if "lora_" in name:
            if diff > 1e-6:
                adapters_changed = True
        else:
            if diff > 1e-6:
                base_changed = True
    return base_changed, adapters_changed
