import time
import numpy as np


def run_greedy_decode(model, tokenizer, prompt, runs=3):
    outputs = []
    for _ in range(runs):
        np.random.seed(42)
        tokens = tokenizer.encode(prompt)
        generated = []
        for _ in range(10):
            logits = model.forward(tokens + generated)
            next_token = int(np.argmax(logits))
            generated.append(next_token)
        outputs.append(tokens + generated)
    return outputs


def measure_latencies(model, tokenizer, prompt):
    cold_start = time.perf_counter_ns()
    tokens = tokenizer.encode(prompt)
    cold_latencies = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        model.forward(tokens)
        cold_latencies.append(time.perf_counter_ns() - t0)

    reused_latencies = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        model.forward_cached(tokens)
        reused_latencies.append(time.perf_counter_ns() - t0)

    return cold_latencies, reused_latencies
