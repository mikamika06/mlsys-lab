## Context

In transformer‑based language models a *key–value (KV) cache* stores the hidden states of past tokens so that subsequent decoding steps can reuse them without recomputation.  
When a batch contains both **prefill** requests (e.g., for a new prompt) and **decode** requests (e.g., for streaming generation), the scheduler must decide which KV slots to read from memory.  The cache is typically organized into *pages* of fixed size; each page holds several contiguous KV slots.

For a given batch we are interested in two quantities:

1. **Total number of KV slots that will be accessed** – this is the cardinality of the union of all requested indices.
2. **Flat gather‑index list** – the exact order in which the cache engine would fetch the slots, preserving the relative order of prefill and decode requests but without duplicates.

Mathematically, if  
\[
P = (p_1,\dots,p_{|P|}) \quad\text{and}\quad D = (d_1,\dots,d_{|D|})
\]
are the sequences of requested indices for prefill and decode respectively, we define
\[
S = \{\, p_i : 1\le i\le |P|\} \cup \{\, d_j : 1\le j\le |D|\,\}
\]
and the gather list \(G\) as the concatenation of the first occurrences of each element in \(P\) followed by the first occurrences in \(D\) that are not already in \(S\).

The page size is provided for realism but does not affect the counting logic in this simplified task.

## Task

Implement `measure_mixed_batch`:

```python
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
```

The function must be pure Python; no external libraries are required.  
It should handle empty lists and duplicate indices correctly.

## Example

```python
prefill = [0, 2, 4]
decode  = [1, 3, 5]
total, gather = measure_mixed_batch(prefill, decode, page_size=8)
print(total)   # 6
print(gather)  # [0, 2, 4, 1, 3, 5]
```

## What the gate checks

The grader verifies that the returned tuple **exactly** matches a reference implementation.  
If either the count or the gather list differs in any element or order, the submission fails.
