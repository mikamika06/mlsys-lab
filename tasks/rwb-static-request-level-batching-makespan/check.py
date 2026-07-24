import numpy as np


def _oracle(output_lens, batch_size):
    counts = []
    for i in range(0, len(output_lens), batch_size):
        chunk = output_lens[i:i + batch_size]
        counts.append(int(np.max(chunk)))
    makespan = int(sum(counts))
    return makespan, counts


def grade(sol, fx) -> dict:
    output_lens = np.asarray(fx["output_lens"], dtype=np.int64)
    batch_size = 4

    ref_makespan, ref_counts = _oracle(output_lens, batch_size)

    try:
        got_makespan, got_counts = sol.static_batching_makespan(output_lens.copy(), batch_size)
        got_makespan = int(got_makespan)
        got_counts = [int(c) for c in got_counts]
    except Exception:
        return {"exact_match": 0.0}

    ok = 1.0 if (got_makespan == ref_makespan and got_counts == ref_counts) else 0.0
    return {"exact_match": ok}
