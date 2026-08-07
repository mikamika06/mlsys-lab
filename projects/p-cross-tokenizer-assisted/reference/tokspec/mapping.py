import numpy as np


def map_tokens_to_bytes(vocab, token_ids):
    res = []
    for tid in token_ids:
        b = vocab.get(tid, b"")
        res.append(b)
    return b"".join(res)
