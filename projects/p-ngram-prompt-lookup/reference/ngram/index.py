class PromptNgramIndex:
    def __init__(self, prompt_tokens, n=4):
        self.n = n
        self.prompt_tokens = list(prompt_tokens)
        self.index = {}
        for i in range(len(self.prompt_tokens) - n + 1):
            ngram = tuple(self.prompt_tokens[i:i + n])
            if ngram not in self.index:
                self.index[ngram] = []
            self.index[ngram].append(i + n)

    def lookup(self, current_tokens):
        if len(current_tokens) < self.n:
            return []
        key = tuple(current_tokens[-self.n:])
        positions = self.index.get(key, [])
        candidates = []
        for pos in positions:
            match_tokens = []
            idx = pos
            while idx < len(self.prompt_tokens):
                match_tokens.append(self.prompt_tokens[idx])
                idx += 1
            candidates.append(match_tokens)
        return candidates
