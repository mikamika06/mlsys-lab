import numpy as np
from mlsys.scorers import argmax_agreement

def _ref(draft_logits: np.ndarray, target_logits: np.ndarray) -> list[int]:
    draft_top = np.argmax(draft_logits, axis=1)
    target_top = np.argmax(target_logits, axis=1)
    return list(np.where(draft_top == target_top, draft_top, target_top))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(5):
        try:
            T = rng.integers(1, 10)
            V = rng.integers(2, 20)
            draft = rng.standard_normal((T, V))
            target = rng.standard_normal((T, V))

            got = sol.greedy_speculative(draft, target)
            if not isinstance(got, (list, np.ndarray)):
                ok = 0.0
                break
            got_list = list(np.asarray(got).astype(int))

            ref = _ref(draft, target)
            if got_list != ref:
                ok = 0.0
                break

            # Build candidate logits for argmax_agreement
            cand_logits = np.zeros_like(target)
            for t, idx in enumerate(got_list):
                cand_logits[t, idx] = 1.0
            agreement = argmax_agreement(target, cand_logits)
            if agreement != 1.0:
                ok = 0.0
                break

        except Exception:
            ok = 0.0
            break
    return {"spec_agreement": ok}
