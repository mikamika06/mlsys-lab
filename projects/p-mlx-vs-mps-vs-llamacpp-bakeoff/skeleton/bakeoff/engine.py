class BaseEngine:
    def load(self, model_path: str, quantization: str):
        raise NotImplementedError

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> dict:
        raise NotImplementedError


class MLXEngine(BaseEngine):
    pass


class MPSEngine(BaseEngine):
    pass


class LlamaCppEngine(BaseEngine):
    pass
