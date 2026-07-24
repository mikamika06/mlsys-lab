import numpy as np


def _oracle(arrivals, prompt_lens, gen_lens, n_prefill, n_decode, t_pre, t_dec):
    """Reference disaggregated-serving simulation: for each request, in
    given order, route its PREFILL phase to whichever prefill worker will
    be free soonest (ties -> lowest worker index), run it (waiting for
    both that worker AND the request's own arrival), then route its
    DECODE phase the same way among decode workers, using the request's
    prefill-completion time as the earliest it can start decoding."""
    prefill_available = [0.0] * n_prefill
    decode_available = [0.0] * n_decode
    prefill_assignments = [[] for _ in range(n_prefill)]
    decode_assignments = [[] for _ in range(n_decode)]

    n = len(arrivals)
    for i in range(n):
        p = min(range(n_prefill), key=lambda w: prefill_available[w])
        start = max(prefill_available[p], arrivals[i])
        finish = start + prompt_lens[i] * t_pre
        prefill_available[p] = finish
        prefill_assignments[p].append(i)

        d = min(range(n_decode), key=lambda w: decode_available[w])
        dstart = max(decode_available[d], finish)
        dfinish = dstart + gen_lens[i] * t_dec
        decode_available[d] = dfinish
        decode_assignments[d].append(i)

    return prefill_assignments, decode_assignments


def _hand_cases():
    cases = []

    cases.append(
        dict(
            arrivals=[0.0, 0.0, 0.0, 0.0],
            prompt_lens=[100, 100, 100, 100],
            gen_lens=[10, 10, 10, 10],
            n_prefill=2,
            n_decode=2,
            t_pre=0.01,
            t_dec=0.02,
        )
    )

    cases.append(
        dict(
            arrivals=[0.0, 1.0, 2.0, 3.0, 4.0],
            prompt_lens=[500, 10, 10, 10, 10],
            gen_lens=[5, 5, 5, 5, 5],
            n_prefill=1,
            n_decode=3,
            t_pre=0.02,
            t_dec=0.05,
        )
    )

    cases.append(
        dict(
            arrivals=[0.0],
            prompt_lens=[50],
            gen_lens=[20],
            n_prefill=3,
            n_decode=3,
            t_pre=0.01,
            t_dec=0.01,
        )
    )

    cases.append(
        dict(
            arrivals=[],
            prompt_lens=[],
            gen_lens=[],
            n_prefill=2,
            n_decode=2,
            t_pre=0.01,
            t_dec=0.01,
        )
    )

    return cases


def _gen_case(rng, n_req=10):
    arrivals = sorted(float(x) for x in rng.uniform(0.0, 50.0, size=n_req))
    prompt_lens = list(rng.integers(10, 2000, size=n_req))
    gen_lens = list(rng.integers(1, 200, size=n_req))
    n_prefill = int(rng.integers(1, 4))
    n_decode = int(rng.integers(1, 4))
    t_pre = float(rng.uniform(0.001, 0.05))
    t_dec = float(rng.uniform(0.005, 0.1))
    return dict(
        arrivals=arrivals, prompt_lens=prompt_lens, gen_lens=gen_lens,
        n_prefill=n_prefill, n_decode=n_decode, t_pre=t_pre, t_dec=t_dec,
    )


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = _hand_cases()
    for _ in range(10):
        cases.append(_gen_case(rng))

    exact = 1.0
    for c in cases:
        ref = _oracle(
            c["arrivals"], c["prompt_lens"], c["gen_lens"],
            c["n_prefill"], c["n_decode"], c["t_pre"], c["t_dec"],
        )
        try:
            got = sol.route_prefill_decode(
                list(c["arrivals"]), list(c["prompt_lens"]), list(c["gen_lens"]),
                c["n_prefill"], c["n_decode"], c["t_pre"], c["t_dec"],
            )
            got_p, got_d = got
            got_p = [list(x) for x in got_p]
            got_d = [list(x) for x in got_d]
        except Exception:
            exact = 0.0
            break
        if (got_p, got_d) != ref:
            exact = 0.0
            break

    return {"exact_match": exact}
