import numpy as np

VOCAB = ["{", "}", '"name"', '"age"', ":", ",", '"john"', '"alice"', "25", "30", '"25"', "\n", "Here is the JSON:", " "]

class MockModel:
    def __init__(self, seed, biases=None):
        self.rng = np.random.RandomState(seed)
        self.vocab_size = len(VOCAB)
        self.biases = biases if biases is not None else np.zeros(self.vocab_size)

    def get_logits(self):
        return self.rng.randn(self.vocab_size) + self.biases

    def sample(self, logits):
        return int(np.argmax(logits))

class MockFSM:
    def __init__(self):
        self.state = 0
        self.transitions = {
            0: [0], 1: [2], 2: [4], 3: [6, 7], 4: [5], 5: [3], 6: [4], 7: [8, 9], 8: [1], 9: []
        }
        self.closing = {
            0: [0, 2, 4, 6, 5, 3, 4, 8, 1],
            1: [2, 4, 6, 5, 3, 4, 8, 1],
            2: [4, 6, 5, 3, 4, 8, 1],
            3: [6, 5, 3, 4, 8, 1],
            4: [5, 3, 4, 8, 1],
            5: [3, 4, 8, 1],
            6: [4, 8, 1],
            7: [8, 1],
            8: [1],
            9: []
        }

    def get_allowed_tokens(self):
        return self.transitions[self.state]

    def advance(self, token_id):
        if token_id not in self.transitions[self.state]:
            raise ValueError(f"Invalid token {token_id} for state {self.state}")
        self.state += 1

    def get_closing_tokens(self):
        return self.closing[self.state]
