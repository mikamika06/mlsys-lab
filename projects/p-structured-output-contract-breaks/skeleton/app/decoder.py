import numpy as np

def apply_grammar_mask(logits: np.ndarray, allowed_tokens: list[int]) -> np.ndarray:
    raise NotImplementedError

def decode_with_schema(model, fsm, max_tokens: int) -> list[int]:
    raise NotImplementedError

def generate_safe(model, fsm, max_tokens: int) -> list[int]:
    raise NotImplementedError
