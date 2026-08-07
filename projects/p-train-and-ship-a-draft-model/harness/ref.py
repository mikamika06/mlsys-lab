import numpy as np

def get_vocab_size():
    return 32

def get_hidden_size():
    return 8

def get_target_probs():
    np.random.seed(1337)
    probs = {}
    vs = get_vocab_size()
    for i in range(vs):
        logits = np.random.randn(vs)
        e = np.exp(logits - np.max(logits))
        probs[i] = e / np.sum(e)
    return probs

def get_dataset():
    np.random.seed(42)
    return np.random.randint(0, get_vocab_size(), 100).tolist()

def oracle_expected_acceptance(p, q):
    return float(np.sum(np.minimum(p, q)))

def oracle_speedup(gamma, alpha, c):
    tokens = (1.0 - alpha**(gamma + 1.0)) / (1.0 - alpha)
    cost = 1.0 + gamma * c
    return tokens / cost
