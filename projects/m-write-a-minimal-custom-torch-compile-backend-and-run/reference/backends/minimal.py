import torch

def register_minimal_backend(model):
    def compiler_fn(gm, sample_inputs):
        return gm.forward
    return torch.compile(model, backend=compiler_fn)
