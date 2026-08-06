import time
import torch

def analyze_latency(model, sample_inputs, mode_config):
    if mode_config is None:
        compiled_model = model
    else:
        compiled_model = torch.compile(model, backend=mode_config["backend"], mode=mode_config["mode"])

    start_compile = time.perf_counter()
    _ = compiled_model(*sample_inputs)
    compile_time = time.perf_counter() - start_compile

    latencies = []
    for _ in range(5):
        start_steady = time.perf_counter()
        _ = compiled_model(*sample_inputs)
        latencies.append(time.perf_counter() - start_steady)

    steady_latency = sum(latencies) / len(latencies)
    return {"compile_time": compile_time, "steady_latency": steady_latency}
