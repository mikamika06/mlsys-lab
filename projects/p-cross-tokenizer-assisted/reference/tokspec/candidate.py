import numpy as np
from tokspec.mapping import map_tokens_to_bytes


def transfer_candidates(draft_tokens, draft_vocab, target_vocab):
    raw_bytes = map_tokens_to_bytes(draft_vocab, draft_tokens)
    target_tokens = []
    curr = 0
    rev_vocab = {v: k for k, v in target_vocab.items()}
    while curr < len(raw_bytes):
        matched = False
        for length in range(min(len(raw_bytes) - curr, 16), 0, -1):
            chunk = raw_bytes[curr:curr + length]
            if chunk in rev_vocab:
                target_tokens.append(rev_vocab[chunk])
                curr += length
                matched = True
                break
        if not matched:
            b = raw_bytes[curr:curr + 1]
            if b in rev_vocab:
                target_tokens.append(rev_vocab[b])
            else:
                target_tokens.append(0)
            curr += 1
    return target_tokens
