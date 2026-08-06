import numpy as np


def generate_test_cases():
    np.random.seed(1337)
    cases = []
    for _ in range(10):
        vocab_size = 32
        p_logits = np.random.randn(vocab_size)
        q_logits = np.random.randn(vocab_size)
        token_id = int(np.random.randint(0, vocab_size))
        temp = float(np.random.choice([0.7, 1.0, 1.2]))
        p_target = float(np.random.choice([0.8, 0.9, 1.0]))
        p_draft = float(np.random.choice([0.8, 0.9, 1.0]))
        cases.append({
            "p_logits": p_logits,
            "q_logits": q_logits,
            "token_id": token_id,
            "temp": temp,
            "p_target": p_target,
            "p_draft": p_draft
        })
    return cases
