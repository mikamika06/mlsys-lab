import numpy as np

class StatefulRunner:
    def __init__(self, contract):
        self.contract = contract
        self.seq_len = 0
        self.states = {}
        for st in contract["states"]:
            self.states[st["name"]] = np.zeros(st["shape"], dtype=np.float32)

    def step(self, token):
        self.seq_len += 1
        logits = np.array([float(token + self.seq_len)], dtype=np.float32)
        return logits, self.seq_len

    def reset(self):
        self.seq_len = 0
        for k in self.states:
            self.states[k].fill(0)
