from typing import List, Tuple

def measure_mixed_batch(prefill: List[int], decode: List[int], page_size: int) -> Tuple[int, List[int]]:
    """
    Compute the total number of unique KV slots that will be read and return a flat
    gather‑index list preserving the order of first appearance.

    Parameters
    ----------
    prefill : List[int]
        Indices requested by the prefill phase.
    decode : List[int]
        Indices requested by the decode phase.
    page_size : int
        Size of a cache page (unused in this simplified implementation).

    Returns
    -------
    Tuple[int, List[int]]
        The first element is the number of unique KV slots accessed.
        The second element is the flat gather‑index list as described above.
    """
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
