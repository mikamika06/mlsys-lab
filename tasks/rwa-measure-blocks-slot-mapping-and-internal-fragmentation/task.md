## Context

In many systems that store a sequence of items in fixed‑size memory blocks, the number of blocks required is given by the ceiling of the ratio between the sequence length and the block size:

$$\text{num\_blocks} = \left\lceil \frac{\text{seq\_len}}{\text{block\_size}}\right\rceil.$$

The last block may contain unused slots, which are called *internal fragmentation* or *waste*. The waste is

$$\text{waste} = \text{num\_blocks}\times\text{block\_size}-\text{seq\_len}.$$

When the blocks are laid out contiguously in memory, each logical token at position $i$ occupies a *physical slot* given by its global index:

$$\text{slot\_mapping}[i] = i,$$

for $0 \le i < \text{seq\_len}$, assuming the first block starts at physical slot 0.

## Task

Implement `measure_blocks(seq_len: int, block_size: int) -> Tuple[int, np.ndarray, int]`:

```python
def measure_blocks(seq_len: int, block_size: int):
    ...
```

The function must return a tuple `(num_blocks, slot_mapping, waste)` where:

* **num_blocks** – the integer number of blocks needed.
* **slot_mapping** – a 1‑D NumPy array of shape `(seq_len,)` containing the global physical slot index for each logical token.
* **waste** – the total number of unused slots in the last block.

All arithmetic should be performed using Python integers and NumPy; no loops are required. The returned `slot_mapping` must use dtype `np.int64`.

## Example

```python
import numpy as np
num_blocks, slot_mapping, waste = measure_blocks(5, 3)
print(num_blocks)      # 2
print(slot_mapping)    # [0 1 2 3 4]
print(waste)           # 1
```

Here `seq_len=5` and `block_size=3`. Two blocks are needed (`ceil(5/3)=2`). The first block holds slots 0–2, the second holds slots 3–5; slot 5 is unused, so waste = 6−5 = 1.

## What the gate checks

The grader computes a reference implementation using NumPy and Python arithmetic. It then compares the candidate’s output exactly:

* `num_blocks` must equal the ceiling division result.
* `waste` must match the computed waste.
* `slot_mapping` must be identical to the array of global slot indices.

All three components are compared with exact equality; any mismatch causes the gate to fail. No performance or style checks are performed for this task.
