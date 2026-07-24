## Context

Consider a chain of $k$ elementwise ops applied in sequence to a tensor of
$n$ elements stored as `dtype_bytes` bytes each — e.g. `y = relu(x * a + b)`
is a chain of $k=3$ elementwise ops (mul, add, relu). Elementwise ops are
memory-bound: computing each output element costs almost nothing compared
to the cost of moving its bytes to/from HBM (off-chip memory).

**Unfused**, each op is its own kernel launch. A kernel launch must read
its input tensor from HBM and write its output tensor back to HBM (it
cannot assume the previous kernel's output is still sitting in registers or
cache once that kernel has exited). For $k$ chained ops, each producing an
$n$-element intermediate, that's $k$ reads and $k$ writes of $n$ elements
each:
$$
\text{bytes\_unfused} = 2 \cdot k \cdot n \cdot \text{dtype\_bytes}.
$$

**Fused**, the compiler emits a *single* kernel for the whole chain: it
reads the original input once, keeps every intermediate value in
registers while it runs all $k$ ops back-to-back on each element, and
writes only the final output once:
$$
\text{bytes\_fused} = 2 \cdot n \cdot \text{dtype\_bytes}.
$$

Notice `bytes_fused` does not depend on $k$ at all — the $k-1$ intermediate
tensors never touch HBM. The traffic-saving ratio of fusion is therefore
$$
\text{ratio} = \frac{\text{bytes\_fused}}{\text{bytes\_unfused}} = \frac{1}{k},
$$
independent of $n$ and `dtype_bytes`: fusing a chain of $k$ elementwise ops
cuts memory traffic to $1/k$ of the unfused cost.

## Task

Implement:

```python
def fusion_traffic(n: int, k: int, dtype_bytes: int) -> tuple[int, int]:
    ...
```

* `n` — number of elements in the tensor.
* `k` — number of chained elementwise ops ($k \ge 1$).
* `dtype_bytes` — bytes per element.

Return `(bytes_unfused, bytes_fused)`:

* `bytes_unfused` — total HBM read+write traffic running the chain as $k$
  separate (unfused) kernel launches: $2 \cdot k \cdot n \cdot \text{dtype\_bytes}$.
* `bytes_fused` — total HBM read+write traffic running the chain as one
  fused kernel: $2 \cdot n \cdot \text{dtype\_bytes}$.

## Example

```python
fusion_traffic(n=1_000_000, k=4, dtype_bytes=4)
# bytes_unfused = 2 * 4 * 1_000_000 * 4 = 32,000,000
# bytes_fused   = 2 *     1_000_000 * 4 =  8,000,000
# -> (32000000, 8000000)   (fused traffic is 1/4 of unfused, matching 1/k)
```

## What the gate checks

Two gates run over several seeded random `(n, k, dtype_bytes)` configs:

* **size_ratio** — the worst-case absolute difference between your
  `bytes_fused / bytes_unfused` and the analytic $1/k$, across all trials,
  must be $\le 10^{-12}$.
* **rel_err** — the worst-case relative error between your returned
  `(bytes_unfused, bytes_fused)` and the exact formulas above, across all
  trials, must be $\le 10^{-12}$. This catches implementations that get the
  *ratio* right by accident (e.g. a shared missing/extra constant factor
  that cancels out of the ratio) but return the wrong absolute byte counts.

Both gates must pass; any exception fails both.
