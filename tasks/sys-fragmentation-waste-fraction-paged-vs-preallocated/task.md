## Context

Memory systems for variable-length sequences often waste capacity when they allocate every
sequence to the maximum observed length. PagedAttention-style allocation avoids this by
splitting storage into fixed-size blocks and allocating only the blocks that are needed.

For a batch of sequence lengths $L_1, L_2, \dots, L_n$ and a page size $B$, a paged
allocator uses

$$
P_{\mathrm{paged}} = \sum_{i=1}^{n} \left\lceil \frac{L_i}{B} \right\rceil B
$$

storage units. The wasted fraction inside allocated pages is

$$
W_{\mathrm{paged}} =
\frac{P_{\mathrm{paged}} - \sum_i L_i}{P_{\mathrm{paged}}}.
$$

A simple preallocated allocator reserves the maximum sequence length for every sequence:

$$
P_{\mathrm{pre}} = n \cdot \max_i(L_i) ,
$$

with wasted fraction

$$
W_{\mathrm{pre}} =
\frac{P_{\mathrm{pre}} - \sum_i L_i}{P_{\mathrm{pre}}}.
$$

The goal is to compute these fragmentation fractions directly from the allocation
model.

## Task

Implement `fragmentation_waste_fraction(lengths, page_size)`:

```python
def fragmentation_waste_fraction(lengths, page_size):
    ...
```

The input `lengths` is a non-empty sequence of positive integer sequence lengths.
`page_size` is a positive integer page size.

Return a tuple:

```python
(paged_fraction, preallocated_fraction)
```

where both values are Python floats. Do not simulate individual tokens or allocate
large buffers. Compute the two fractions from the formulas above.

## Example

```python
result = fragmentation_waste_fraction([10, 17, 33], 16)

# Returns approximately:
# (0.21428571428571427, 0.4)
```

The paged allocation is $16 + 32 + 48 = 96$ units, with $60$ useful units.
The preallocated allocation is $3 \cdot 33 = 99$ units.

## What the gate checks

The gate computes the expected fractions using the same allocation equations as a
reference oracle and compares both returned values. The reported `size_ratio` metric
is the maximum absolute difference between the implementation output and the oracle
output. It must satisfy $size\_ratio \le 10^{-9}$.
