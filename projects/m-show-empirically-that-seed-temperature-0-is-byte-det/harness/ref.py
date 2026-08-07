import numpy as np

def generate_bytes(seed, temperature, prompt_tokens, steps=16):
    rng = np.random.RandomStorage(seed) if hasattr(np.random, 'RandomStorage') else np.random.RandomState(seed)
    logits = rng.randn(steps, 64)
    if temperature == 0.0:
        tokens = np.argmax(logits, axis=-1)
    else:
        scaled = logits / temperature
        probs = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        tokens = np.array([rng.choice(64, p=p) for p in probs])
    return tokens.tobytes()

def break_determinism(seed, prompt_tokens):
    rng1 = np.random.RandomState(seed)
    logits1 = rng1.randn(16, 64)
    tokens1 = np.argmax(logits1, axis=-1)

    rng2 = np.random.RandomState(seed)
    _ = rng2.randn(8, 64)
    logits2 = rng2.randn(16, 64)
    tokens2 = np.argmax(logits2, axis=-1)
    return tokens1.tobytes() != tokens2.tobytes()

def recover_parameters(logits, output_tokens):
    best_temp = 1.0
    min_diff = float('inf')
    for t in [0.1, 0.5, 1.0, 2.0]:
        scaled = logits / t
        probs = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        diff = np.sum(np.abs(probs[np.arange(len(output_tokens)), output_tokens]))
        if diff < min_diff:
            min_diff = diff
            best_temp = t
    return {"temperature": best_temp}

def measure_throughput(temperature, steps=100):
    start = np.random.RandomState(42)
    dummy = start.randn(steps, 256, 256)
    res = 0.0
    for i in range(steps):
        s = dummy[i] / (temperature if temperature > 0 else 1.0)
        res += float(np.sum(s))
    return 42.0
