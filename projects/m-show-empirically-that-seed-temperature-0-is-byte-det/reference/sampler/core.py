import numpy as np


def verify_determinism(logits, seed):
    rng1 = np.random.default_rng(seed)
    t1 = np.argmax(logits, axis=-1)
    rng2 = np.random.default_rng(seed)
    t2 = np.argmax(logits, axis=-1)
    return np.array_equal(t1, t2)


def detect_fracture(logits, seed):
    rng1 = np.random.default_rng(seed)
    _ = rng1.random()
    t1 = np.argmax(logits, axis=-1)
    rng2 = np.random.default_rng(seed)
    t2 = np.argmax(logits, axis=-1)
    return not np.array_equal(t1, t2)
