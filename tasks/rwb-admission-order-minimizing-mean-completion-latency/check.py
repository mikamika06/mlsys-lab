import numpy as np


def _simulate(lengths_in_order, slot_count) -> np.ndarray:
    slot_free = np.zeros(slot_count, dtype=np.float64)
    completions = np.empty(len(lengths_in_order), dtype=np.float64)
    for i, length in enumerate(lengths_in_order):
        j = int(np.argmin(slot_free))
        c = slot_free[j] + float(length)
        slot_free[j] = c
        completions[i] = c
    return completions


def _ref_optimal_mean(gen_lens, slot_count) -> float:
    order = sorted(range(len(gen_lens)), key=lambda i: gen_lens[i])
    comps = _simulate([gen_lens[i] for i in order], slot_count)
    return float(np.mean(comps))


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    # hand-picked small cases
    scenarios.append(([3, 1, 4, 1, 5], 2))
    scenarios.append(([7], 1))
    scenarios.append(([7], 5))
    scenarios.append(([5, 5, 5, 5, 5, 5], 6))  # slots >= N, ties
    scenarios.append(([9, 2, 7, 2, 9, 2, 7, 4], 1))  # single slot, ties
    scenarios.append(([9, 2, 7, 2, 9, 2, 7, 4], 3))

    # seeded random scenarios
    for n, s in [(10, 3), (15, 4), (25, 5), (30, 1), (12, 12)]:
        gen_lens = rng.integers(1, 50, size=n).tolist()
        scenarios.append((gen_lens, s))

    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0

    for gen_lens, slot_count in _scenarios():
        n = len(gen_lens)
        opt_mean = _ref_optimal_mean(gen_lens, slot_count)

        try:
            order, claimed_mean = sol.admission_order(list(gen_lens), slot_count)
        except Exception:
            return {"rel_err": float("inf")}

        try:
            order = [int(x) for x in order]
            claimed_mean = float(claimed_mean)
        except Exception:
            return {"rel_err": float("inf")}

        if sorted(order) != list(range(n)):
            return {"rel_err": float("inf")}

        actual_comps = _simulate([gen_lens[i] for i in order], slot_count)
        actual_mean = float(np.mean(actual_comps))

        # claimed value must match what the returned order actually achieves
        report_tol = 1e-6 * max(1.0, abs(actual_mean))
        if abs(actual_mean - claimed_mean) > report_tol:
            return {"rel_err": float("inf")}

        err = abs(actual_mean - opt_mean) / (abs(opt_mean) + 1e-12)
        worst = max(worst, err)

    return {"rel_err": worst}
