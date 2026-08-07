class EagleSampler:
    def __init__(self, topk=4):
        raise NotImplementedError

    def sample_tree(self, logits, temperature=1.0):
        raise NotImplementedError
