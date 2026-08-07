class Tokenizer:
    def __init__(self):
        raise NotImplementedError

    def encode(self, text, special=True):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError
