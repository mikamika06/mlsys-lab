## Context

The **roofline model** predicts the best performance a kernel can reach on a machine
described by two numbers: a peak compute rate $P$ (FLOP/s) and a peak DRAM bandwidth
$B$ (bytes/s).

For a kernel that performs $W$ floating-point operations while moving $Q$ bytes
between DRAM and the chip, the *arithmetic intensity* is

$$
I \;=\; \frac{W}{Q} \qquad \text{[FLOP/byte]} .
$$

A kernel can never run faster than compute allows, and never faster than the memory
system can feed it, so the attainable rate is

$$
A(I) \;=\; \min\bigl(P,\; I \cdot B\bigr) \qquad \text{[FLOP/s]} .
$$

The two roofs cross at the **ridge point**

$$
I^{*} \;=\; \frac{P}{B},
$$

the smallest intensity at which a kernel can still be compute-bound. Kernels with
$I < I^{*}$ are memory-bound; the machine's FLOP peak is irrelevant for them.

## Task

Implement `roofline_attainable(flops, bytes_moved, peak_flops, bandwidth)`:

```python
def roofline_attainable(flops, bytes_moved, peak_flops, bandwidth):
    ...
```

* `flops` — 1-D array of $W_k$, the FLOP count of kernel $k$.
* `bytes_moved` — 1-D array of $Q_k$, the bytes kernel $k$ moves (same length).
* `peak_flops` — scalar $P$ in FLOP/s.
* `bandwidth` — scalar $B$ in bytes/s.

Return the triple

```python
(ai, attainable, ridge)
```

* `ai` — float64 array of arithmetic intensities $I_k = W_k / Q_k$,
* `attainable` — float64 array of $A(I_k) = \min(P,\; I_k B)$ in FLOP/s,
* `ridge` — the scalar ridge point $I^{*} = P / B$ in FLOP/byte.

## Example

```python
import numpy as np

flops  = np.array([2e9, 8e9])       # 2 and 8 GFLOP of work
bytes_ = np.array([1e9, 1e8])       # 1 GB and 100 MB moved
P, B   = 4e12, 2e11                 # 4 TFLOP/s, 200 GB/s

ai, att, ridge = roofline_attainable(flops, bytes_, P, B)
print(ai)     # [ 2. 80.]      FLOP/byte
print(att)    # [4.0e+11 4.0e+12]   first kernel memory-bound, second compute-bound
print(ridge)  # 20.0           FLOP/byte
```

The first kernel has $I = 2 < 20$, so it is memory-bound and stuck at
$I B = 400$ GFLOP/s — only 10 % of the machine's peak. The second one sits above the
ridge and saturates the compute roof.

## What the gate checks

All three outputs are compared against a NumPy reference recomputed from the same
randomly generated machines and kernels (seeded, so the run is deterministic). The
fixtures deliberately straddle the ridge point, so a solution that forgets the
`min(...)` clamp — or clamps on the wrong side — is memory-bound-wrong on some
kernels and compute-bound-wrong on others.

Each of `ai_rel_err`, `attainable_rel_err` and `ridge_rel_err` is a global relative
L2 error and must be $< 10^{-6}$.
