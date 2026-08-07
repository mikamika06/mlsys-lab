import numpy as np


def _oracle(K, Q, prompt_len, budget, recent_window):
    d = K.shape[1]
    T = Q.shape[0]
    resident = list(range(prompt_len))
    score = {i: 0.0 for i in resident}

    trajectory = []
    for t in range(T):
        q = Q[t]
        idx = sorted(resident)
        Kc = K[idx]
        logits = (q @ Kc.T) / np.sqrt(d)
        logits = logits - np.max(logits)
        w = np.exp(logits)
        w = w / np.sum(w)
        for j, i in enumerate(idx):
            score[i] += float(w[j])

        new_pos = prompt_len + t
        resident.append(new_pos)
        score[new_pos] = 0.0

        if len(resident) > budget:
            current = sorted(resident)
            protected = set(current[-recent_window:]) if recent_window > 0 else set()
            evictable = [i for i in current if i not in protected]
            worst = min(evictable, key=lambda i: (score[i], i))
            resident.remove(worst)
            del score[worst]

        trajectory.append(sorted(resident))

    return trajectory


def _cases():
    rng = np.random.default_rng(13)
    cases = []
    for _ in range(6):
        d = int(rng.integers(3, 8))
        prompt_len = int(rng.integers(2, 8))
        budget = prompt_len + int(rng.integers(1, 5))
        recent_window = int(rng.integers(1, min(budget, 4) + 1))
        T = int(rng.integers(6, 16))
        K = rng.standard_normal((prompt_len + T, d))
        Q = rng.standard_normal((T, d))
        cases.append((K, Q, prompt_len, budget, recent_window))
    return cases


def grade(sol, fx) -> dict:
    total = 0
    correct = 0
    for K, Q, prompt_len, budget, recent_window in _cases():
        ref = _oracle(K, Q, prompt_len, budget, recent_window)
        try:
            got = sol.h2o_eviction_trajectory(K.tolist(), Q.tolist(), prompt_len, budget, recent_window)
        except Exception:
            total += len(ref)
            continue

        try:
            if len(got) != len(ref):
                total += len(ref)
                continue
            for g, r in zip(got, ref):
                total += 1
                if len(g) <= budget and sorted(int(x) for x in g) == r:
                    correct += 1
        except Exception:
            total += len(ref)
            continue

    return {"exact_match": (correct / total) if total else 0.0}
