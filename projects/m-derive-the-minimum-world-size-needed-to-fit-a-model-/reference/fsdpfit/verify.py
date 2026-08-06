import torch

def verify_model_weights(original_model, sharded_model):
    orig_state = original_model.state_dict()
    shrd_state = sharded_model.state_dict()
    for name, param in orig_state.items():
        if name not in shrd_state:
            return False
        if not torch.allclose(param, shrd_state[name], atol=1e-5, rtol=1e-5):
            return False
    return True
