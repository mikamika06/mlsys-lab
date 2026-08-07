class GGUFTokenizer:
    """GGUF tokenizer engine."""
    def __init__(self, vocab_data):
        self.vocab_data = vocab_data

    def tokenize(self, text):
        return [ord(c) for c in text]

    def detokenize(self, tokens):
        return "".join(chr(t) for t in tokens)
