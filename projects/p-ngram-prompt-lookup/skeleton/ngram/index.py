class PromptNgramIndex:
    def __init__(self, prompt_tokens, n=4):
        raise NotImplementedError

    def lookup(self, current_tokens):
        raise NotImplementedError
