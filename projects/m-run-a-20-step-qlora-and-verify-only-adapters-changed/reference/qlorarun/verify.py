import torch


def verify_only_adapters_changed(initial_state, final_state, model):
    base_unchanged = True
    adapters_changed = False

    for name, param in model.named_parameters():
        if name in initial_state:
            init_p = initial_state[name]
            final_p = param.data
            if "lora_" in name:
                if not torch.equal(init_p, final_p):
                    adapters_changed = True
            else:
                if not torch.equal(init_p, final_p):
                    base_unchanged = False

    return base_unchanged and adapters_changed
