## Context

The roofline model bounds the throughput a kernel can attain on a given
machine by the lesser of the machine's peak compute rate and the rate at
which its memory bandwidth can feed that compute:

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}, \qquad
\text{attainable}(\text{kernel}) = \min\!\big(\pi,\ \text{AI}\cdot\beta\big),
$$

where $\pi$ is the machine's peak FLOP/s, $\beta$ is its peak memory
bandwidth (bytes/s), and $\text{AI}$ is the kernel's arithmetic intensity
(FLOPs per byte moved). A kernel with low arithmetic intensity is
**bandwidth-bound** — it never reaches $\pi$ no matter how efficient its
compute is — while a kernel with high arithmetic intensity is
**compute-bound** and attains (up to) $\pi$.

Given a machine and a batch of kernels, ranking the kernels by their
*attainable* throughput (not their raw FLOP count, and not their
arithmetic intensity alone) tells you which operations the roofline model
predicts will actually run fastest on that hardware.

## Task

Implement `rank_kernels_by_throughput`:

```python
def rank_kernels_by_throughput(kernels: list[tuple[float, float]], peak_flops: float, peak_bw: float) -> list[int]:
    ...
```

* `kernels` — a list of `(flops, bytes_moved)` pairs, one per kernel.
* `peak_flops` — the machine's peak compute rate $\pi$ (FLOP/s).
* `peak_bw` — the machine's peak memory bandwidth $\beta$ (bytes/s).

For each kernel $i$, compute $\text{AI}_i = \text{flops}_i / \text{bytes}_i$
and its attainable throughput $\min(\pi, \text{AI}_i \cdot \beta)$.

Return a list of the kernel **indices** `0..len(kernels)-1`, sorted so
that the kernel with the **highest** attainable throughput comes first
(descending order). Break ties by original index, ascending (i.e. the
sort must be stable with respect to the input order).

## Example

```python
kernels = [
    (1e9, 1e9),    # AI = 1     -> bandwidth-bound
    (8e9, 1e6),    # AI = 8000  -> compute-bound, hits peak
    (4e9, 1e7),    # AI = 400   -> compute-bound, hits peak (tie with #1)
]
peak_flops = 2e12   # 2 TFLOP/s
peak_bw = 4e11       # 400 GB/s

rank_kernels_by_throughput(kernels, peak_flops, peak_bw)
# attainable: [4e11, 2e12, 2e12]
# -> [1, 2, 0]   (kernels 1 and 2 tie at peak_flops; index 1 < 2, so 1 first)
```

## What the gate checks

The grader builds several deterministic `(kernels, peak_flops, peak_bw)`
cases — including cases with exact throughput ties — computes the
reference attainable throughput for every kernel with the formula above,
and derives the ground-truth descending, stably tie-broken ordering.
**exact_match** is `1.0` only if your returned index list matches the
reference ordering exactly, for every case; otherwise it is `0.0`.
