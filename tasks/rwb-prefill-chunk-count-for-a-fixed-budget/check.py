import numpy as np


def _oracle(prompt_lens, chunk_budget):
    n = np.asarray(prompt_lens, dtype=np.int64)
    num_chunks = -(-n // chunk_budget)  # ceil division
    last_chunk = n - (num_chunks - 1) * chunk_budget
    return num_chunks, last_chunk


def grade(sol, fx) -> dict:
    prompt_lens = np.asarray(fx["prompt_lens"], dtype=np.int64)

    ok = 1.0
    for chunk_budget in (1, 128, 256, 512, 4096):
        ref_num, ref_last = _oracle(prompt_lens, chunk_budget)
        try:
            got = sol.chunk_counts(prompt_lens.copy(), chunk_budget)
            got_num = np.asarray(got["num_chunks"], dtype=np.int64)
            got_last = np.asarray(got["last_chunk"], dtype=np.int64)
        except Exception:
            ok = 0.0
            break

        if got_num.shape != ref_num.shape or got_last.shape != ref_last.shape:
            ok = 0.0
            break
        if not np.array_equal(got_num, ref_num) or not np.array_equal(got_last, ref_last):
            ok = 0.0
            break

    return {"exact_match": ok}
