class ConstrainedEngine:
    def __init__(self, vocab: list, schema: dict):
        raise NotImplementedError

    def generate(self, logits_generator, num_predict: int) -> dict:
        raise NotImplementedError
