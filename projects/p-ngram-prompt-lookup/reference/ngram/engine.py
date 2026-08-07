from ngram.index import PromptNgramIndex
from ngram.policy import select_candidates, should_disable

class NgramSpeculativeEngine:
    def __init__(self, prompt_tokens, k=4, disable_threshold=0.1):
        self.prompt_tokens = list(prompt_tokens)
        self.k = k
        self.disable_threshold = disable_threshold
        self.index = PromptNgramIndex(prompt_tokens)
        self.history = []
        self.disabled = False
        self.accepted_count = 0
        self.total_proposals = 0

    def step(self, target_model_func, current_output):
        if self.disabled:
            token = target_model_func(current_output)
            return [token], 1, 0.0

        full_context = self.prompt_tokens + current_output
        candidates = select_candidates(self.index, full_context, self.k)
        if not candidates:
            token = target_model_func(current_output)
            self.history.append(0.0)
            if should_disable(self.history, self.disable_threshold):
                self.disabled = True
            return [token], 1, 0.0

        speculative = candidates[:self.k]
        verified, accepted_num = target_model_func(current_output, speculative)

        self.total_proposals += len(speculative)
        self.accepted_count += accepted_num
        acc_rate = accepted_num / max(1, len(speculative))
        self.history.append(acc_rate)

        if should_disable(self.history, self.disable_threshold):
            self.disabled = True

        return verified, len(verified), acc_rate

    def run(self, target_model_func, max_steps=50):
        current_output = []
        steps = 0
        while steps < max_steps:
            tokens, _, _ = self.step(target_model_func, current_output)
            current_output.extend(tokens)
            steps += 1
            if len(tokens) == 0:
                break
        return current_output
