class VocabHandler:
    """Handles vocabulary loading and mapping."""
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges

    def process(self, text):
        return [ord(c) for c in text]
