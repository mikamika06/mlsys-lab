import torch


class MockPEFTModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(32, 32)
        self.lora_A = torch.nn.Linear(32, 8, bias=False)
        self.lora_B = torch.nn.Linear(8, 32, bias=False)

    def forward(self, x):
        out = self.base_layer(x)
        if x.size(1) > 16:
            torch._dynamo.graph_break()
        return out + self.lora_B(self.lora_A(x))


def get_oracle_model_and_inputs():
    torch.manual_seed(42)
    model = MockPEFTModule()
    x_base = torch.randn(2, 8, 32)
    x_varied = torch.randn(2, 24, 32)
    return model, x_base, x_varied
