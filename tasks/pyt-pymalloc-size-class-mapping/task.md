## Context

CPython's `pymalloc` allocator handles small object allocations without going
back to the system `malloc` for every request. Memory is organized into
**arenas** (large chunks obtained from the OS), each split into **pools**
(fixed-size pages), each pool subdivided into **blocks** of a single fixed
size. Every pool serves exactly one *size class* — a fixed block size — so
allocating a small object means finding the pool for its size class and
handing out one free block, no general-purpose allocation search required.

pymalloc defines its size classes with two constants: an alignment of
$\text{ALIGN} = 8$ bytes, and a threshold of $\text{THRESHOLD} = 512$ bytes.
For a requested allocation of $n$ bytes with $1 \le n \le \text{THRESHOLD}$,
the request is rounded **up** to the next multiple of $8$, and the size-class
index is

$$
\mathrm{idx}(n) = \left\lceil \frac{n}{8} \right\rceil - 1 = \left\lfloor \frac{n-1}{8} \right\rfloor, \qquad 0 \le \mathrm{idx}(n) \le 63 .
$$

This gives $\text{THRESHOLD} / \text{ALIGN} = 64$ size classes: class $0$
covers requests of $1$–$8$ bytes, class $1$ covers $9$–$16$ bytes, and so on
up to class $63$ covering $505$–$512$ bytes. Requests larger than the
threshold ($n > 512$) are too big for pymalloc's pool machinery and are
delegated straight to the system allocator — they have no size class.

## Task

Implement `pymalloc_size_class(sizes)`:

```python
def pymalloc_size_class(sizes: list[int]) -> list[int]:
    ...
```

`sizes` is a list of positive integers (requested byte counts, each
$\ge 1$). For each size $n$, return its pymalloc size-class index using the
formula above if $1 \le n \le 512$, or the sentinel $-1$ if $n > 512$
(delegated to the system allocator). Return a list of `int` the same length
as `sizes`, in the same order.

## Example

```python
pymalloc_size_class([1, 8, 9, 512, 513, 1000])
# [0, 0, 1, 63, -1, -1]
```

`1` and `8` both round up to the 8-byte block, class `0`. `9` needs the next
block, class `1`. `512` is exactly the last small-object size, class `63`.
`513` and `1000` exceed the `512`-byte threshold, so both map to `-1`.

## What the gate checks

The grader builds a fixture of requested sizes — boundary values around every
multiple of $8$ up to and past $512$, plus randomly sampled sizes — and
computes the reference size-class index for each with the formula above. Your
returned vector must equal the reference vector exactly (`exact_match`
$= 1.0$). An off-by-one in the rounding (e.g. using `n // 8` instead of
`(n - 1) // 8`) shifts every boundary size into the wrong class and fails the
gate.
