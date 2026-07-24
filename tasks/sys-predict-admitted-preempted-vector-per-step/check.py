import numpy as np


def _reference(arrivals, lengths, priorities, budget, num_steps):
    """Real oracle: simulate a preemptive-priority continuous-batching
    scheduler with a fixed running-set capacity `budget`.

    At every step, the running set is recomputed from scratch as the
    `budget` best-priority requests among everyone who has arrived and has
    not yet finished (lower `priorities` value == higher priority, ties
    broken by request id). Anyone newly in that set was "admitted" this
    step; anyone who was running last step, still has work left, but isn't
    in the new set was "preempted" this step. Requests that simply finish
    (their remaining length hits zero) are neither -- they just vanish
    from the active pool on their own.
    """
    n = len(arrivals)
    remaining = list(lengths)
    prev_running = set()
    steps = []

    for t in range(num_steps):
        active_pool = [i for i in range(n) if arrivals[i] <= t and remaining[i] > 0]
        ordered = sorted(active_pool, key=lambda i: (priorities[i], i))
        new_running = set(ordered[:budget])

        admitted = sorted(new_running - prev_running)
        preempted = sorted((prev_running & set(active_pool)) - new_running)
        steps.append({"admitted": admitted, "preempted": preempted})

        for i in new_running:
            remaining[i] -= 1
        prev_running = new_running

    return steps


def _cases():
    return [
        dict(
            arrivals=[0, 0, 1, 2, 2, 3],
            lengths=[3, 2, 4, 1, 2, 5],
            priorities=[2, 1, 0, 3, 0, 4],
            budget=2,
            num_steps=8,
        ),
        dict(
            arrivals=[0, 1, 1, 2, 4, 5, 5],
            lengths=[2, 3, 1, 2, 4, 1, 2],
            priorities=[5, 4, 3, 2, 0, 1, 0],
            budget=3,
            num_steps=9,
        ),
        dict(
            arrivals=[0, 0, 0, 0],
            lengths=[1, 1, 1, 1],
            priorities=[3, 2, 1, 0],
            budget=1,
            num_steps=6,
        ),
        dict(
            arrivals=[0, 2, 3, 3, 6],
            lengths=[6, 2, 3, 3, 1],
            priorities=[1, 1, 0, 2, 0],
            budget=2,
            num_steps=10,
        ),
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for c in _cases():
        expected = _reference(c["arrivals"], c["lengths"], c["priorities"], c["budget"], c["num_steps"])
        try:
            got = sol.schedule_admit_preempt(
                list(c["arrivals"]), list(c["lengths"]), list(c["priorities"]),
                c["budget"], c["num_steps"],
            )
            got = list(got)
        except Exception:
            ok = 0.0
            break

        if len(got) != len(expected):
            ok = 0.0
            break

        for got_step, exp_step in zip(got, expected):
            try:
                g_adm = sorted(int(x) for x in got_step["admitted"])
                g_pre = sorted(int(x) for x in got_step["preempted"])
            except Exception:
                ok = 0.0
                break
            if g_adm != exp_step["admitted"] or g_pre != exp_step["preempted"]:
                ok = 0.0
                break
        if ok == 0.0:
            break

    return {"exact_match": ok}
