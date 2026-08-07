class TokenizerConverter:
    """Converts tokenizer artifacts."""
    def __init__(self, raw_vocab, raw_merges):
        self.raw_vocab = raw_vocab
        self.raw_merges = raw_merges

    def convert(self):
        return {"vocab": self.raw_vocab, "merges": self.raw_merges}
