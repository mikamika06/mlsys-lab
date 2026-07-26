## Context

When allocating a collection of variable‑length sequences into fixed‑size memory blocks, two common strategies are used:

* **Paged allocation** – each sequence is placed in its own set of blocks.  
  For a sequence of length $\ell$ and block size $b$, the number of blocks required is
  $$\left\lceil \frac{\ell}{b}\right\rceil,$$
  so the total number of blocks for all sequences is the sum over all lengths.

* **Contiguous pre‑allocation** – a single large contiguous region is reserved that can hold the longest sequence.  
  If $L_{\max}$ is the maximum length, each sequence is allocated a block of size
  $\left\lceil \frac{L_{\max}}{b}\right\rceil b$, and the total number of blocks is
  $$n_{\text{seq}}\;\left\lceil \frac{L_{\max}}{b}\right\rceil,$$
  where $n_{\text{seq}}$ is the number of sequences.

The **wasted slots** for a strategy are the difference between the total memory reserved and the sum of all sequence lengths:
$$
\text{wasted} = \text{blocks}\times b - \sum_i \ell_i .
$$

These metrics are used in production libraries to decide whether to use paged or contiguous allocation.

## Task

Implement `wasted_slots(lengths, bs)`:

```python
def wasted_slots(lengths: list[int], bs: int) -> dict[str, tuple[int,int]]:
    ...
```

* `lengths` – a list of positive integers representing the lengths of each sequence.
* `bs` – the block size (positive integer).

The function must return a dictionary with two keys:

* `"paged"` – a tuple `(num_blocks_paged, wasted_paged)`.
* `"contiguous"` – a tuple `(num_blocks_contig, wasted_contig)`.

Both tuples contain integers. The calculation should follow exactly the formulas described in the context section.

## Example

```python
>>> lengths = [5, 7, 12]
>>> bs = 8
>>> wasted_slots(lengths, bs)
{
    'paged': (4, 8),        # ceil(5/8)+ceil(7/8)+ceil(12/8) = 1+1+2 = 4 blocks; 4*8 - 24 = 8 wasted
    'contiguous': (6, 24)   # ceil(12/8) = 2 blocks each, 3*2 = 6 blocks; 6*8 - 24 = 24 wasted
}
```

Contiguous pre-allocation sizes every sequence to the longest one in the batch, which is why it
wastes three times what paged allocation does on this input.

## What the gate checks

The grader computes a reference result using NumPy for a set of random test cases and compares it to the candidate’s output. The metric `exact_match` is used: the candidate passes only if all returned tuples match the oracle exactly.
