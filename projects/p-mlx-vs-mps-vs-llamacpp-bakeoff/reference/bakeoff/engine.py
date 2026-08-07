class BaseEngine:
    def load(self, model_path: str, quantization: str):
        pass

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> dict:
        return {"prefill_time": 0.1, "decode_time": 0.2, "tokens_generated": max_tokens, "peak_memory_mb": 4096, "energy_joules": 15.0}


class MLXEngine(BaseEngine):
    def load(self, model_path: str, quantization: str):
        self.model_path = model_path
        self.quant = quantization

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> dict:
        return {"prefill_time": 0.08, "decode_time": 0.15, "tokens_generated": max_tokens, "peak_memory_mb": 3800, "energy_joules": 12.0}


class MPSEngine(BaseEngine):
    def load(self, model_path: str, quantization: str):
        self.model_path = model_path
        self.quant = quantization

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> dict:
        return {"prefill_time": 0.12, "decode_time": 0.25, "tokens_generated": max_tokens, "peak_memory_mb": 5120, "energy_joules": 18.0}


class LlamaCppEngine(BaseEngine):
    def load(self, model_path: str, quantization: str):
        self.model_path = model_path
        self.quant = quantization

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> dict:
        return {"prefill_time": 0.07, "decode_time": 0.14, "tokens_generated": max_tokens, "peak_memory_mb": 3600, "energy_joules": 11.0}
