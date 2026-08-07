import numpy as np

def align_bytes(draft_tokens, target_vocab_bytes):
    out = []
    for tok in draft_tokens:
        b = bytes([tok % 256])
        matched = -1
        for idx, vb in enumerate(target_vocab_bytes):
            if vb == b:
                matched = idx
                break
        out.append(matched)
    return out
