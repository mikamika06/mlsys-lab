import json
import torch

def serialize_quant_state(quant_state):
    state_dict = {}
    for k, v in quant_state.__dict__.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.detach().cpu().tolist()
        elif isinstance(v, (str, int, float, bool, type(None))):
            state_dict[k] = v
        elif isinstance(v, tuple):
            state_dict[k] = list(v)
        else:
            try:
                state_dict[k] = json.dumps(v)
            except Exception:
                pass
    return json.dumps(state_dict)
