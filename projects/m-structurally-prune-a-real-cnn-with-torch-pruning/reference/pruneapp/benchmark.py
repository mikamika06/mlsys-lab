import torch
import time


def measure_speedup(original_model, pruned_model, example_inputs, warmup=5, iters=20):
    original_model.eval()
    pruned_model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = original_model(example_inputs)

        start = time.time()
        for _ in range(iters):
            _ = original_model(example_inputs)
        orig_time = time.time() - start

        for _ in range(warmup):
            _ = pruned_model(example_inputs)

        start = time.time()
        for _ in range(iters):
            _ = pruned_model(example_inputs)
        pruned_time = time.time() - start

    speedup = orig_time / pruned_time if pruned_time > 0 else 1.0
    return orig_time, pruned_time, speedup
