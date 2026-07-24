import numpy as np


def _oracle(starts, durations, num_tokens, kv_bytes_per_token):
    """Sweep-line over the half-open transfer windows [start, start+dur).
    At a shared timestamp, END events are applied before START events, so
    a transfer that finishes exactly when another begins is correctly
    treated as not overlapping it."""
    events = []
    for s, d, n in zip(starts, durations, num_tokens):
        rate = (n * kv_bytes_per_token) / d
        events.append((s, 1, rate))       # start: applied after end at tie
        events.append((s + d, 0, -rate))  # end: applied first at tie
    events.sort(key=lambda e: (e[0], e[1]))

    cur = 0.0
    peak = 0.0
    for _t, _typ, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _hand_cases():
    cases = []

    # Two disjoint transfers -> peak is just the larger single rate.
    cases.append(
        dict(
            starts=[0.0, 5.0],
            durations=[1.0, 1.0],
            num_tokens=[100, 400],
            kv=2.0,
        )
    )

    # Fully overlapping transfers -> peak is the sum of all rates.
    cases.append(
        dict(
            starts=[0.0, 0.0, 0.0],
            durations=[2.0, 2.0, 2.0],
            num_tokens=[100, 200, 50],
            kv=4.0,
        )
    )

    # Staggered starts, partial overlap.
    cases.append(
        dict(
            starts=[0.0, 0.5, 1.0, 1.2],
            durations=[1.0, 1.0, 1.0, 0.5],
            num_tokens=[128, 256, 64, 512],
            kv=2.0,
        )
    )

    # Exact touch: one ends exactly when the next starts -> must NOT
    # count as overlapping (half-open window).
    cases.append(
        dict(
            starts=[0.0, 1.0],
            durations=[1.0, 1.0],
            num_tokens=[1000, 1000],
            kv=1.0,
        )
    )

    # Single request.
    cases.append(dict(starts=[3.0], durations=[0.25], num_tokens=[2048], kv=2.0))

    return cases


def _gen_case(rng, n=6):
    starts = list(rng.uniform(0.0, 10.0, size=n))
    durations = list(rng.uniform(0.05, 2.0, size=n))
    num_tokens = list(rng.integers(16, 4096, size=n))
    kv = float(rng.uniform(0.5, 8.0))
    return dict(starts=starts, durations=durations, num_tokens=num_tokens, kv=kv)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(8):
        cases.append(_gen_case(rng))

    worst = 0.0
    for c in cases:
        ref = _oracle(c["starts"], c["durations"], c["num_tokens"], c["kv"])
        try:
            got = float(
                sol.peak_kv_transfer_bandwidth(
                    list(c["starts"]), list(c["num_tokens"]),
                    list(c["durations"]), c["kv"],
                )
            )
        except Exception:
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)

    return {"rel_err": worst}
