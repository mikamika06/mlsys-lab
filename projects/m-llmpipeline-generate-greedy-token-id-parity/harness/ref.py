import numpy as np
import time

class MockModel:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def infer(self, input_ids):
        time.sleep(0.005)
        seq_len = len(input_ids)
        logits = np.zeros((1, seq_len, self.vocab_size))
        for i in range(seq_len):
            next_tok = (input_ids[i] * 7 + 13) % self.vocab_size
            logits[0, i, next_tok] = 10.0
        return logits

class MockPipeline:
    def __init__(self, model, has_chat=True):
        self.model = model
        self.has_chat = has_chat

    def generate(self, prompt_ids, max_tokens):
        time.sleep(0.005)
        prompt_ids = list(prompt_ids)
        out = []
        for _ in range(max_tokens):
            next_tok = (prompt_ids[-1] * 7 + 13) % self.model.vocab_size
            out.append(next_tok)
            prompt_ids.append(next_tok)
        return out

    def apply_chat_template(self, messages):
        if not self.has_chat:
            raise RuntimeError("Exception: Chat template is missing")
        return "[CHAT] " + " | ".join(m["content"] for m in messages)
