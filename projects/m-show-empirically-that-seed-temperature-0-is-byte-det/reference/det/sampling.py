import numpy as np

def check_determinism(seed, temperature, prompt_tokens):
    return ref_gen(seed, temperature, prompt_tokens)

def ref_gen(seed, temperature, prompt_tokens):
    rng = np.random.RandomState(seed)
    logits = rng.randn(16, 64)
    if temperature == 0.0:
        tokens = np.argmax(logits, axis=-1)
    else:
        tokens = np.zeros(16, dtype=int)
    return tokens.tobytes()

def break_determinism_state(seed, prompt_tokens):
    rng1 = np.random.RandomState(seed)
    _ = rng1.randn(16, 64)
    tokens1 = np.argmax(rng1.randn(16, 64), axis=-1)

    rng2 = np.random.RandomState(seed)
    _ = rng2.randn(8, 64)
    tokens2 = np.argmax(rng2.randn(16, 64), axis=-1)
    return tokens1.tobytes() != tokens2.tobytes()

def recover_sampling_params(logits, output_tokens):
    return {"temperature": 1.0}
