import numpy as np


def _oracle(reqs, N):
    n_reqs = len(reqs)
    admission_order = sorted(range(n_reqs), key=lambda i: (reqs[i][0], i))

    slots = [None] * N
    waiting = []
    admit_ptr = 0

    trajectory = []
    t = 0
    safety = sum(g for _, g in reqs) + n_reqs + N + 10
    while t < safety:
        for s in range(N):
            occ = slots[s]
            if occ is not None and occ["progress"] >= reqs[occ["req"]][1]:
                slots[s] = None

        if admit_ptr >= n_reqs and not waiting and all(s is None for s in slots):
            break

        while admit_ptr < n_reqs and reqs[admission_order[admit_ptr]][0] <= t:
            waiting.append(admission_order[admit_ptr])
            admit_ptr += 1

        for s in range(N):
            if slots[s] is None and waiting:
                slots[s] = {"req": waiting.pop(0), "progress": 0}

        for s in range(N):
            if slots[s] is not None:
                slots[s]["progress"] += 1

        trajectory.append([slots[s]["req"] if slots[s] is not None else -1 for s in range(N)])
        t += 1

    return trajectory


def _synthetic_cases():
    rng = np.random.default_rng(59)
    cases = []
    for _ in range(4):
        n_reqs = int(rng.integers(4, 14))
        arrivals = np.sort(rng.integers(0, 8, size=n_reqs)).tolist()
        gen_lens = rng.integers(1, 6, size=n_reqs).tolist()
        reqs = list(zip(arrivals, gen_lens))
        N = int(rng.integers(1, 5))
        cases.append((reqs, N))
    return cases


def grade(sol, fx) -> dict:
    fixture_reqs = list(zip(fx["arrivals"].tolist(), fx["gen_lens"].tolist()))
    cases = [(fixture_reqs, 3)] + _synthetic_cases()

    total = 0
    correct = 0
    for reqs, N in cases:
        ref = _oracle(reqs, N)
        total += len(ref)
        try:
            got = sol.slot_occupancy_trajectory(list(reqs), N)
        except Exception:
            continue

        try:
            for k in range(min(len(got), len(ref))):
                row = [int(x) for x in got[k]]
                if row == ref[k]:
                    correct += 1
        except Exception:
            pass

    return {"exact_match": (correct / total) if total else 0.0}
