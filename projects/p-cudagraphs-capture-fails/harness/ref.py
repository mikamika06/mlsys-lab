import torch

def get_reference_model(fixed=False):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)
            self.fixed = fixed

        def forward(self, x):
            if not self.fixed:
                temp = torch.empty(x.size(0), 16, device=x.device)
                temp.copy_(x)
                return self.linear(temp)
            else:
                return self.linear(x)

    return MockModel()
