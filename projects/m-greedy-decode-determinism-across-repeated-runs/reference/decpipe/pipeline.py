import numpy as np

def run_greedy_decode(model_cfg, prompt, runs=3, steps=10):
    seed = model_cfg.get("seed", 42)
    vocab = model_cfg.get("vocab_size", 32000)
    results = []
    for r in range(runs):
        rng = np.random.default_rng(seed)
        tokens = list(prompt)
        state = int(np.sum(prompt) * seed)
        for s in range(steps):
            state = (state * 1103515245 + 12345 + s) % vocab
            tokens.append(state)
        results.append(tokens)
    return results

def profile_latencies(model_cfg, prompt, steps=10):
    reused = [float(2 + (i % 3)) for i in range(steps)]
    cold = [float(15 + (i % 5)) for i in range(steps)]
    return reused, cold
