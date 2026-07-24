import numpy as np


def _sim_static(gen_lens, batch_size) -> float:
    total = 0.0
    n = len(gen_lens)
    for i in range(0, n, batch_size):
        batch = gen_lens[i:i + batch_size]
        total += max(batch)
    return float(total)


def _sim_continuous(gen_lens, batch_size) -> float:
    slot_free = np.zeros(batch_size, dtype=np.float64)
    for length in gen_lens:
        j = int(np.argmin(slot_free))
        slot_free[j] = slot_free[j] + float(length)
    return float(np.max(slot_free))


def _ref(gen_lens, batch_size):
    ms = _sim_static(gen_lens, batch_size)
    mc = _sim_continuous(gen_lens, batch_size)
    ratio = ms / mc
    return ms, mc, ratio


def _rel(ref_v, got_v) -> float:
    return abs(got_v - ref_v) / (abs(ref_v) + 1e-12)


def _scenarios():
    scenarios = []

    scenarios.append(([1, 1, 1, 10], 2))
    scenarios.append(([1, 1, 1, 1, 1, 1, 1], 1))     # batch_size=1 -> ratio must be 1.0
    scenarios.append(([3, 7, 2, 9, 1, 4], 6))         # batch_size=N -> ratio must be 1.0
    scenarios.append(([5, 5, 5, 5, 5], 3))            # uneven last batch
    scenarios.append(([50, 1, 1, 1, 1, 1, 1, 1], 4))  # one long request skews static badly
    scenarios.append(([1] * 9 + [30], 3))

    rng = np.random.default_rng(0)
    for n, bs in [(10, 3), (20, 4), (15, 1), (15, 15), (33, 7)]:
        gen_lens = rng.integers(1, 40, size=n).tolist()
        scenarios.append((gen_lens, bs))

    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for gen_lens, batch_size in _scenarios():
        ms_ref, mc_ref, ratio_ref = _ref(gen_lens, batch_size)

        try:
            ms_got, mc_got, ratio_got = sol.compare_batching_throughput(list(gen_lens), batch_size)
        except Exception:
            return {"rel_err": float("inf")}

        try:
            ms_got = float(ms_got)
            mc_got = float(mc_got)
            ratio_got = float(ratio_got)
        except Exception:
            return {"rel_err": float("inf")}

        for ref_v, got_v in ((ms_ref, ms_got), (mc_ref, mc_got), (ratio_ref, ratio_got)):
            e = _rel(ref_v, got_v)
            if not np.isfinite(e):
                return {"rel_err": float("inf")}
            worst = max(worst, e)

    return {"rel_err": worst}
