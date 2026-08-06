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
