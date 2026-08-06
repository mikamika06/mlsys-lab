import time
import torch


def ablate_backend_ladder(model, example_inputs, warmups=2, runs=5):
    """Run backend ladder ablation on a given CPU model and return relative metrics."""
    backends = ["eager", "aot_eager", "inductor"]
    results = {}
    
    for b in backends:
        if b == "eager":
            compiled = model
        else:
            compiled = torch.compile(model, backend=b)
            
        for _ in range(warmups):
            _ = compiled(*example_inputs)
            
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = compiled(*example_inputs)
        t1 = time.perf_counter()
        
        results[b] = (t1 - t0) / runs

    baseline = results["eager"]
    output = {}
    for b in backends:
        output[b] = {
            "latency": results[b],
            "speedup": baseline / results[b] if results[b] > 0 else 1.0
        }
    return output
