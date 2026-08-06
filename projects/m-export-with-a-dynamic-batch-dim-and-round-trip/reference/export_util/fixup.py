import torch

def fix_data_dependent_flow(model):
    class WrappedModel(torch.nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig

        def forward(self, x):
            return self.orig(x)

    return WrappedModel(model)
