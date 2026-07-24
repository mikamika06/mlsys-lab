import numpy as np


def _oracle_simulate(arrival_iters, gen_lens, max_num_seqs):
    n = len(arrival_iters)
    waiting = sorted(range(n), key=lambda i: (arrival_iters[i], i))
    active = {}
    result = []
    t = 0
    while True:
        while waiting and len(active) < max_num_seqs and arrival_iters[waiting[0]] <= t:
            rid = waiting.pop(0)
            active[rid] = 0
        if not active and not waiting:
            break
        result.append(sorted(active.keys()))
        for rid in list(active.keys()):
            active[rid] += 1
        for rid in list(active.keys()):
            if active[rid] >= gen_lens[rid]:
                del active[rid]
        t += 1
    return result


def grade(sol, fx) -> dict:
    arrival_iters = np.asarray(fx["arrival_iters"], dtype=np.int64)
    gen_lens = np.asarray(fx["gen_lens"], dtype=np.int64)
    run_id = np.asarray(fx["run_id"], dtype=np.int64)
    max_num_seqs = np.asarray(fx["max_num_seqs"], dtype=np.int64)

    ok = 1.0
    for r in range(max_num_seqs.shape[0]):
        mask = run_id == r
        a = arrival_iters[mask]
        g = gen_lens[mask]
        cap = int(max_num_seqs[r])

        expected = _oracle_simulate(a.tolist(), g.tolist(), cap)
        try:
            got = sol.simulate_active_set(a.copy(), g.copy(), cap)
            got_norm = [sorted(int(x) for x in step) for step in got]
        except Exception:
            ok = 0.0
            break

        if got_norm != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
