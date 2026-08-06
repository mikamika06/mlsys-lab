import torch

def verify_adapter_only(base_state, post_state):
    for name in base_state:
        if "adapter" not in name and "lora" not in name:
            if not torch.equal(base_state[name], post_state[name]):
                return False
    return True
