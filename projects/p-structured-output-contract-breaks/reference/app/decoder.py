import numpy as np

def apply_grammar_mask(logits: np.ndarray, allowed_tokens: list[int]) -> np.ndarray:
    mask = np.full_like(logits, -np.inf)
    for t in allowed_tokens:
        mask[t] = logits[t]
    return mask

def decode_with_schema(model, fsm, max_tokens: int) -> list[int]:
    tokens = []
    while fsm.state != 9 and len(tokens) < max_tokens:
        logits = model.get_logits()
        allowed = fsm.get_allowed_tokens()
        masked = apply_grammar_mask(logits, allowed)
        token_id = model.sample(masked)
        fsm.advance(token_id)
        tokens.append(token_id)
    return tokens

def generate_safe(model, fsm, max_tokens: int) -> list[int]:
    tokens = []
    while fsm.state != 9:
        closing = fsm.get_closing_tokens()
        if len(tokens) + len(closing) >= max_tokens:
            tokens.extend(closing)
            for t in closing:
                fsm.advance(t)
            break

        logits = model.get_logits()
        allowed = fsm.get_allowed_tokens()
        masked = apply_grammar_mask(logits, allowed)
        token_id = model.sample(masked)
        fsm.advance(token_id)
        tokens.append(token_id)
    return tokens
