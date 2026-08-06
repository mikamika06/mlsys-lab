import torch.nn as nn

def fix_control_flow(model):
    class FixedModule(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.linear = m.linear
        def forward(self, x):
            mask = (x.sum() > 0).float()
            out1 = self.linear(x)
            out2 = self.linear(x) * -1.0
            return out1 * mask + out2 * (1.0 - mask)
    return FixedModule(model)
