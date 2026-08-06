import numpy as np


def _kv(token):
    t = float(token)
    return np.array([t, t * 0.5 + 1.0], dtype=np.float64), np.array([t * 2.0 - 1.0], dtype=np.float64)


def _decode(k_cache, v_cache, token):
    score = 0.0
    for row in k_cache:
        for val in row:
            score += float(val)
    for row in v_cache:
        for val in row:
            score += float(val)
    score += float(token)
    return int(score) % 100


def chunked_prefill_decode(prompt, chunk_sizes, decode_tokens):
    k_cache = []
    v_cache = []
    emitted = []
    pos = 0
    decode_pos = 0

    for chunk_index, size in enumerate(chunk_sizes):
        for token in prompt[pos:pos + size]:
            k, v = _kv(int(token))
            k_cache.append(k)
            v_cache.append(v)
        pos += size

        if chunk_index != len(chunk_sizes) - 1:
            emitted.append(_decode(k_cache, v_cache, int(decode_tokens[decode_pos])))
            decode_pos += 1

    return (
        np.stack(k_cache, axis=0),
        np.stack(v_cache, axis=0),
        np.array(emitted, dtype=np.int64),
    )
