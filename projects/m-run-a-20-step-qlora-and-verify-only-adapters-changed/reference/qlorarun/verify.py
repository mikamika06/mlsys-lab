import torch


def verify_parameter_updates(initial_state, final_state, model):
    base_unchanged = True
    adapters_changed = False
    for name, param in model.named_parameters():
        if name in initial_state:
            init_val = initial_state[name]
            final_val = param.detach().cpu()
            if "lora_" in name:
                if not torch.equal(init_val, final_val):
                    adapters_changed = True
            else:
                if not torch.equal(init_val, final_val):
                    base_unchanged = False
    return {"base_unchanged": base_unchanged, "adapters_changed": adapters_changed}
