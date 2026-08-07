class Tokenizer:
    def __init__(self):
        self.special_tokens = {"<|endoftext|>": 2, "<|beginoftext|>": 1}

    def encode(self, text, special=True):
        res = [ord(c) for c in text]
        if special:
            res = [self.special_tokens["<|beginoftext|>"]] + res + [self.special_tokens["<|endoftext|>"]]
        return res

    def decode(self, tokens):
        filtered = [t for t in tokens if t not in self.special_tokens.values()]
        return "".join([chr(t) if 32 <= t <= 126 else "?" for t in filtered])
