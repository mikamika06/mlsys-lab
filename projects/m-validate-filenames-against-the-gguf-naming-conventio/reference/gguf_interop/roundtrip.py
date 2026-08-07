import torch

def gguf_to_torch(tensors):
    out = {}
    for k, v in tensors.items():
        out[k] = v.clone()
    return out

def torch_to_gguf(tensors):
    out = {}
    for k, v in tensors.items():
        out[k] = v.clone()
    return out
