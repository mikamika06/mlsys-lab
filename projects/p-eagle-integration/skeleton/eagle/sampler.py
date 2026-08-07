class DraftSampler:
    def __init__(self, temperature=1.0):
        raise NotImplementedError

    def sample(self, logits):
        raise NotImplementedError
