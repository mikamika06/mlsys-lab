class DraftModel:
    def __init__(self, vocab_size: int, hidden_size: int):
        raise NotImplementedError

    def forward(self, token: int):
        raise NotImplementedError

    def get_probs(self, token: int):
        raise NotImplementedError
