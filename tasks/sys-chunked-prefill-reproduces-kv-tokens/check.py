import numpy as np


def _kv(token):
    t = float(token)
    return np.array([t, t * 0.5 + 1.0], dtype=np.float64), np.array([t * 2.0 - 1.0], dtype=np.float64)


def _decode(k_cache, v_cache, token):
    score = float(np.sum(k_cache) + np.sum(v_cache) + token)
    return int(score) % 100


def _oracle(prompt, chunk_sizes, decode_tokens):
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
            k_arr = np.stack(k_cache, axis=0)
            v_arr = np.stack(v_cache, axis=0)
            emitted.append(_decode(k_arr, v_arr, int(decode_tokens[decode_pos])))
            decode_pos += 1

    return (
        np.stack(k_cache, axis=0),
        np.stack(v_cache, axis=0),
        np.array(emitted, dtype=np.int64),
    )


def grade(sol, fx) -> dict:
    cases = [
        (np.array([2, 5, 3, 7]), [2, 2], [9]),
        (np.array([1, 4, 8, 2, 6]), [1, 2, 2], [3, 11]),
        (np.array([9, 0, 5, 4, 3, 8]), [3, 1, 2], [7, 12]),
    ]

    ok = 1.0
    for prompt, chunks, decodes in cases:
        try:
            got = sol.chunked_prefill_decode(prompt.copy(), list(chunks), list(decodes))
            ref = _oracle(prompt, chunks, decodes)
            if len(got) != 3:
                ok = 0.0
                break
            if not np.array_equal(np.asarray(got[0]), ref[0]):
                ok = 0.0
                break
            if not np.array_equal(np.asarray(got[1]), ref[1]):
                ok = 0.0
                break
            if not np.array_equal(np.asarray(got[2]), ref[2]):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
