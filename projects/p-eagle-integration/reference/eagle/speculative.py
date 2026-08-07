import numpy as np


class SpeculativeEngine:
    def __init__(self, target_model, draft_head, vocab_size):
        self.target_model = target_model
        self.draft_head = draft_head
        self.vocab_size = vocab_size

    def generate_draft(self, hidden_states, k):
        tokens = []
        curr_h = hidden_states
        for _ in range(k):
            logits = self.draft_head.forward(curr_h)
            token = int(np.argmax(logits))
            tokens.append(token)
            curr_h = hidden_states + 0.01 * token
        return tokens

    def verify_and_sample(self, target_logits, draft_logits, draft_tokens, temperature):
        accepted = []
        if temperature > 0:
            t_probs = np.exp(target_logits / temperature)
            t_probs /= np.sum(t_probs)
            d_probs = np.exp(draft_logits / temperature)
            d_probs /= np.sum(d_probs)
        else:
            t_probs = np.zeros_like(target_logits)
            t_probs[np.argmax(target_logits)] = 1.0
            d_probs = np.zeros_like(draft_logits)
            d_probs[np.argmax(draft_logits)] = 1.0

        for tok in draft_tokens:
            ratio = t_probs[tok] / max(1e-8, d_probs[tok])
            if np.random.rand() < min(1.0, ratio):
                accepted.append(tok)
            else:
                break
        return accepted

    def compute_memory(self, separate_draft_params, head_params):
        return separate_draft_params > head_params

    def measure_speedup(self, base_time, speculative_time):
        speedup = base_time / max(1e-6, speculative_time)
        return speedup > 1.2
