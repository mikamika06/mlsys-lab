import numpy as np


class EagleEngine:
    def __init__(self, target_model_dim, vocab_size, draft_head):
        self.dim = target_model_dim
        self.vocab = vocab_size
        self.head = draft_head

    def forward_target(self, x):
        rng = np.random.default_rng(sum(x))
        hidden = rng.normal(0, 1, (len(x), self.dim))
        logits = rng.normal(0, 1, (len(x), self.vocab))
        return hidden, logits

    def generate_draft(self, hidden):
        logits = self.head.forward(hidden)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        return [int(np.argmax(p)) for p in probs]

    def verify(self, draft_tokens, target_logits, temperature=1.0):
        accepted = []
        for dt, logits in zip(draft_tokens, target_logits):
            scaled = logits / max(temperature, 1e-5)
            probs = np.exp(scaled - np.max(scaled))
            probs /= np.sum(probs)
            target_token = int(np.argmax(probs))
            if dt == target_token:
                accepted.append(dt)
            else:
                accepted.append(target_token)
                break
        return accepted

    def memory_usage_bytes(self, separate_draft_params=100_000_000):
        head_params = self.head.w.size + self.head.b.size
        return {
            "head_bytes": head_params * 4,
            "separate_model_bytes": separate_draft_params * 4
        }
