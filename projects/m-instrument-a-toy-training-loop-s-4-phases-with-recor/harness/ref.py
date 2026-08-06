import time
import torch


def synthetic_step_fn():
    x = torch.randn(100, 100)
    w = torch.randn(100, 100, requires_grad=True)

    with torch.autograd.profiler.record_function("forward"):
        y = torch.matmul(x, w)
        time.sleep(0.001)

    with torch.autograd.profiler.record_function("loss"):
        loss = y.sum()
        time.sleep(0.0005)

    with torch.autograd.profiler.record_function("backward"):
        loss.backward()
        time.sleep(0.0015)

    with torch.autograd.profiler.record_function("optimizer"):
        with torch.no_grad():
            w -= 0.01 * w.grad
            w.grad.zero_()
        time.sleep(0.0005)

    time.sleep(0.001)


def generate_trace_cases():
    return [
        (
            [
                {"type": "push", "name": "A"},
                {"type": "push", "name": "B"},
                {"type": "pop", "name": "B"},
                {"type": "pop", "name": "A"},
            ],
            {"balanced": True, "error_index": -1},
        ),
        (
            [
                {"type": "push", "name": "A"},
                {"type": "push", "name": "B"},
                {"type": "pop", "name": "A"},
            ],
            {"balanced": False, "error_index": 2},
        ),
        (
            [
                {"type": "pop", "name": "A"},
            ],
            {"balanced": False, "error_index": 0},
        ),
        (
            [
                {"type": "push", "name": "A"},
            ],
            {"balanced": False, "error_index": 0},
        ),
    ]
