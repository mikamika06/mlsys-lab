class NgramSpeculativeEngine:
    def __init__(self, prompt_tokens, k=4, disable_threshold=0.1):
        raise NotImplementedError

    def step(self, target_model, generated_so_far):
        raise NotImplementedError

    def run(self, target_model, max_steps=100):
        raise NotImplementedError
