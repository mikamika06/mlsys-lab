import sys
from typing import List, Set

def _oracle(num_layers: int, num_steps: int) -> List[Set[int]]:
    """Reference implementation of the resident‑set scheduler."""
    res: List[Set[int]] = []
    if num_layers == 0:
        return res
    resident: Set[int] = set()
    for step in range(num_steps):
        cur = step % num_layers
        if step == 0:
            resident.add(cur)
            if num_layers > 1:
                resident.add((cur + 1) % num_layers)
        # Emit the resident set at the start of this step
        res.append(set(resident))
        # Evict current layer after it has been used
        resident.discard(cur)
        # Prefetch the layer two steps ahead for the next step
        nxt = (cur + 2) % num_layers
        resident.add(nxt)
    return res

def grade(sol, fx) -> dict:
    """Grade candidate against reference and peak‑resident constraint."""
    cases = [
        (3, 5),
        (4, 10),
        (1, 7),
        (2, 3),
        (5, 12),
    ]
    ok = 1.0
    peak = 0
    for num_layers, num_steps in cases:
        try:
            got: List[Set[int]] = sol.emit_resident_set_per_step(num_layers, num_steps)
        except Exception:
            return {"exact_match": 0.0, "peak_resident": float("inf")}
        ref = _oracle(num_layers, num_steps)
        if got != ref:
            ok = 0.0
        peak = max(peak, max((len(s) for s in got), default=0))
    peak_ok = 1.0 if peak <= 2 else 0.0
    return {"exact_match": ok, "peak_resident": peak_ok}
