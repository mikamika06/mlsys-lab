import torch

class FP32FunctionalWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=False):
            args_fp32 = tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)
            return self.module(*args_fp32, **kwargs)

def wrap_sensitive_modules(model):
    for name, child in model.named_children():
        if isinstance(child, (torch.nn.LayerNorm, torch.nn.Softmax, torch.nn.MSELoss)):
            setattr(model, name, FP32FunctionalWrapper(child))
        else:
            wrap_sensitive_modules(child)
    return model
