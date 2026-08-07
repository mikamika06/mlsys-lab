class GGUFTokenizer:
    """GGUF tokenizer engine."""
    def __init__(self, vocab_data):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError
