class GGUFEngine:
    def __init__(self, importer, tokenizer):
        raise NotImplementedError

    def generate(self, messages: list) -> str:
        raise NotImplementedError
