## Context

When an LLM inference engine stores key-value states, it must decide how to
allocate GPU memory for each sequence in a batch. Two common strategies have
very different **internal fragmentation** profiles.

**Paged allocation.** Memory is divided into fixed-size blocks of $B$ slots.
A sequence of length $\ell$ requires $\lceil \ell / B \rceil$ blocks, consuming
$\lceil \ell / B \rceil \cdot B$ slots total. The wasted (unused) slots are

$$w_{\text{paged}}(\ell) \;=\; \left\lceil \frac{\ell}{B} \right\rceil \cdot B \;-\; \ell.$$

Over a batch of $n$ sequences with lengths $\ell_1, \dots, \ell_n$, the total
paged waste is $\displaystyle\sum_{i=1}^{n} w_{\text{paged}}(\ell_i)$.

**Contiguous pre-allocation.** Each sequence is given a single contiguous
region of $L$ slots, where $L$ is the maximum allowed sequence length. The
wasted slots per sequence are

$$w_{\text{contig}}(\ell) \;=\; L - \ell,$$

and the total contig waste is $\displaystyle\sum_{i=1}^{n} w_{\text{contig}}(\ell_i)$.

Paged allocation generally wastes far fewer slots because the block size $B$
is small (e.g.\ 16 or 64 tokens), while $L$ is the full context window
(e.g.\ 4096). Quantifying this gap is essential when choosing an allocation
strategy for a production serving system.

## Task

Implement `internal_fragmentation`:

```python
def internal_fragmentation(seqlens: list[int], block_size: int, max_len: int) -> tuple[int, int]:
    """Return (paged_waste, contig_waste) as a tuple of two numbers.

    seqlens  – 1-D Python integer array of sequence lengths
    block_size – B, the size of one page/block
    max_len    – L, the maximum sequence length
    """
```

The function receives a list of floats `seqlens` of $n$ positive integer
lengths, an integer block size $B$, and an integer maximum length $L$. It must
return a 2-tuple `(paged_waste, contig_waste)` where each element is a scalar
equal to the total internal fragmentation under the respective strategy.

Use integer or floating-point arithmetic — either is fine — but the results
must be numerically exact (no floating-point rounding error).

## Example

```python
seqlens = [3, 7, 13]
paged_waste, contig_waste = internal_fragmentation(seqlens, block_size=5, max_len=16)
# paged_waste = (5-3) + (10-7) + (15-13) = 2+3+2 = 7
# contig_waste = (16-3) + (16-7) + (16-13) = 13+9+3 = 25
assert paged_waste == 7
assert contig_waste == 25
```

## What the gate checks

Two independent gates. A Python oracle recomputes both waste totals from first
principles. For each scalar the relative error is

$$\text{rel\_err} = \frac{|v_{\text{got}} - v_{\text{ref}}|}{|v_{\text{ref}}| + \epsilon}$$

with $\epsilon = 10^{-12}$. The gate for `paged_rel_err` requires
$\text{rel\_err} < 10^{-9}$, and likewise for `contig_rel_err`. Multiple test
cases with different batch sizes, block sizes, and max lengths are evaluated; all
must pass.
