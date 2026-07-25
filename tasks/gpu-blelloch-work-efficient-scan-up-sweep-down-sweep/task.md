## Context

A naive parallel prefix sum (Hillis-Steele) does `n * log2(n)` additions —
more total work than the `n - 1` additions a sequential scan needs. The
**Blelloch (work-efficient) scan** gets back down to `O(n)` total work by
building the sum as a balanced binary tree over shared memory, in two
phases:

- **Up-sweep (reduce)**: like a parallel reduction, but every internal
  node's partial sum is *kept* instead of thrown away. After this phase,
  the last element holds the total sum of everything.
- **Down-sweep**: walk the same tree back down. Clear the root (last
  element) to the identity (`0`), then at each level swap a node's value
  into its right child while passing the old value left — which
  propagates, level by level, into an **exclusive** prefix sum: `out[i]`
  ends up holding the sum of everything *before* index `i`, not
  including it.

## Task

Implement

```cuda
__global__ void scan(float* out, const float* in, int n);
```

for `n = 32`, one block of `32` threads. Load `in[tid]` into
`__shared__ float temp[32]`, `__syncthreads()`, then:

1. **Up-sweep**: for `d = n/2, n/4, ..., 1` (halving each step,
   `offset` starting at `1` and doubling after every step),
   `__syncthreads()` first, then the first `d` threads do
   `temp[bi] += temp[ai]` where `ai = offset*(2*tid+1)-1`,
   `bi = offset*(2*tid+2)-1`.
2. Thread `0` sets `temp[n-1] = 0`.
3. **Down-sweep**: for `d = 1, 2, ..., n/2` (doubling each step, halving
   `offset` first), `__syncthreads()`, then the first `d` threads swap:
   `t = temp[ai]; temp[ai] = temp[bi]; temp[bi] += t;` (same `ai`, `bi`
   formulas as above, with that step's `offset`).
4. `__syncthreads()`, then `out[tid] = temp[tid]`.

## Example

`in = [3, 1, 4, 1]` (`n = 4`): up-sweep gives `temp = [3, 4, 4, 9]` (pairs
summed, then the whole-array sum lands in `temp[3]`); clearing the root
gives `[3, 4, 4, 0]`; down-sweep produces the exclusive scan
`out = [0, 3, 4, 8]` — `out[i]` is the sum of every element before `i`.

## What the gate checks

`check.py` seeds a fixed random 32-element input, parses `solve.cu` with
the CUDA-C frontend, and launches `scan` as one block of 32 threads on
the software GPU. It compares the resulting `out` array against a numpy
oracle (`np.cumsum` shifted by one — computed from the same seeded input,
never hardcoded) and requires

$$
\mathrm{max\_abs\_err} = \max_i |\text{out}_i - \text{oracle}_i| \le 10^{-6}
$$

An empty kernel body leaves `out` all zeros, which matches the oracle
only at index `0` (whose correct exclusive-scan value genuinely is `0`)
and misses badly everywhere else — `max_abs_err` comes out around `5`,
far past the gate.
