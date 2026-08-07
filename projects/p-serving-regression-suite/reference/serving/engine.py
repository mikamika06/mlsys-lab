import random

class Engine:
    def __init__(self, seed=42):
        self.seed = seed
        self.vocab = {"hello": 1, "world": 2, "<s>": 3, "</s>": 4, "the": 5, "api": 6, "is": 7, "good": 8}
        self.rev = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        res = [self.vocab["<s>"]]
        for w in text.split():
            res.append(self.vocab.get(w, 0))
        return res

    def generate(self, prompt, max_tokens=10, stop_tokens=None, seed=None):
        if stop_tokens is None:
            stop_tokens = []
        eff_seed = seed if seed is not None else self.seed
        rng = random.Random(eff_seed + len(prompt))

        out = []
        for _ in range(max_tokens):
            t = rng.choice(list(self.vocab.values()))
            out.append(t)
            if t in stop_tokens:
                break

        text = " ".join(self.rev.get(t, "<unk>") for t in out)
        return {
            "prompt": prompt,
            "tokens": out,
            "text": text,
            "usage": {"prompt_tokens": len(self.tokenize(prompt)), "completion_tokens": len(out)}
        }
