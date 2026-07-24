from typing import List, Tuple

def _reference(prefill: List[int], decode: List[int], page_size: int) -> Tuple[int, List[int]]:
    seen = set()
    gather: List[int] = []
    for idx in prefill:
        if idx not in seen:
            gather.append(idx)
            seen.add(idx)
    for idx in decode:
        if idx not in seen:
            gather.append(idx)
            seen.add(idx)
    return len(seen), gather

def grade(sol, fx) -> dict:
    cases = [
        ([0, 2, 4], [1, 3, 5], 8),
        ([1, 1, 2], [2, 3], 4),
        ([], [0, 0, 0], 16),
        ([5], [], 8),
        ([0, 1, 2], [2, 1, 0], 32),
    ]
    ok = 1.0
    for prefill, decode, page in cases:
        try:
            got = sol.measure_mixed_batch(prefill, decode, page)
            exp = _reference(prefill, decode, page)
        except Exception:
            return {"exact_match": 0.0}
        if got != exp:
            ok = 0.0
            break
    return {"exact_match": ok}
