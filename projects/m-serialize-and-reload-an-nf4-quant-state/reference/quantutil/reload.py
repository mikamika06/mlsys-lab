import json
import torch

class DummyQuantState:
    pass

def reload_quant_state(serialized):
    data = json.loads(serialized)
    qs = DummyQuantState()
    for k, v in data.items():
        if isinstance(v, list) and k in ["absmax", "quant_state", "code", "offset"]:
            setattr(qs, k, torch.tensor(v))
        else:
            setattr(qs, k, v)
    return qs
