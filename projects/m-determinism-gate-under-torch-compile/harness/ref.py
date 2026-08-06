import torch


def check_determinism(model, inputs, num_runs=5):
    outputs = []
    for _ in range(num_runs):
        with torch.no_grad():
            outputs.append(model(*inputs))
    first = outputs[0]
    for out in outputs[1:]:
        if not torch.equal(first, out):
            return False
    return True


def stabilized_gate(model, inputs, warmup_runs=2, test_runs=3):
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(*inputs)
    return check_determinism(model, inputs, num_runs=test_runs)


MODELS = [
    (torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU()), (torch.randn(4, 16),)),
    (torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.GELU()), (torch.randn(2, 32),)),
]
