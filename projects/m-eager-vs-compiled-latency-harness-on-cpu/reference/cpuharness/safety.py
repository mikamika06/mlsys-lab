import torch

def verify_fullgraph_capture(model, inputs):
    try:
        compiled = torch.compile(model, fullgraph=True)
        _ = compiled(*inputs)
        return None
    except Exception as e:
        return type(e)
