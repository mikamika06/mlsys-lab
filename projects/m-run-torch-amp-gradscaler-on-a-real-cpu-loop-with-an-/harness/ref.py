import torch

def get_test_fixture():
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    stream = [(torch.randn(2, 4, generator=torch.Generator().manual_seed(i)),
               torch.randn(2, 2, generator=torch.Generator().manual_seed(i + 10)))
              for i in range(5)]
    return model, opt, stream
