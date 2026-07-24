## Context

The roofline model says a kernel's achievable throughput is capped by
$\min(\text{peak\_flops}, \text{AI} \times \text{peak\_bandwidth})$, where
**arithmetic intensity** $\text{AI} = \text{FLOPs} / \text{bytes moved}$.
The **ridge point** $\text{peak\_gflops} / \text{peak\_gbps}$ (FLOP/byte)
splits the plot in two: below it a kernel is **memory-bound** (more
bandwidth would help, more compute wouldn't), at or above it the kernel is
**compute-bound**.

Chaining several small elementwise ops — `y = op3(op2(op1(x)))` — as
**separate** passes means every op reads its input from memory and writes
its output back before the next op starts: each op independently pays for
a full read + write of the array. **Fusing** the chain into one pass keeps
every intermediate value in a register: only the very first input is ever
read from memory, and only the very last output is ever written. The
total FLOPs don't change — the total bytes moved shrinks by (roughly) a
factor of how many ops got fused — so fusing an op chain **raises AI
without changing the machine's ridge point**, and can flip a kernel from
memory-bound to compute-bound.

## Task

Implement

```cpp
FusionResult fusion_ai_and_flip(long n, int num_ops, double flops_per_op,
                                 int elem_bytes, double peak_gflops, double peak_gbps);
```

for a chain of `num_ops` elementwise ops (`flops_per_op` FLOPs each) over
`n` elements of `elem_bytes` bytes:

$$
\text{total\_flops} = n \cdot \text{num\_ops} \cdot \text{flops\_per\_op}
$$
$$
\text{unfused\_bytes} = 2 \cdot n \cdot \text{num\_ops} \cdot \text{elem\_bytes}, \qquad
\text{fused\_bytes} = 2 \cdot n \cdot \text{elem\_bytes}
$$
$$
\text{unfused\_ai} = \frac{\text{total\_flops}}{\text{unfused\_bytes}}, \qquad
\text{fused\_ai} = \frac{\text{total\_flops}}{\text{fused\_bytes}}
$$

Compare **both** AIs against the same ridge point
$\text{ridge} = \text{peak\_gflops} / \text{peak\_gbps}$:
`*_compute_bound = *_ai >= ridge` (compute-bound is defined as **at or
above** the ridge, not strictly above), and
`regime_flipped = unfused_compute_bound != fused_compute_bound`.

## Example

`n=100000, num_ops=3, flops_per_op=8, elem_bytes=4, peak_gflops=200,
peak_gbps=100`: `total_flops = 2.4e6`, `unfused_bytes = 9.6e6` so
`unfused_ai = 0.25 * 4 = 1.0`, `fused_bytes = 3.2e6` so
`fused_ai = 3.0`. `ridge = 200/100 = 2.0`. `unfused_ai=1.0 < 2.0`
(memory-bound); `fused_ai=3.0 >= 2.0` (compute-bound) — fusing 3 ops of 8
FLOPs each flips the regime.

## What the gate checks

`main.cpp` runs 6 fixed scenarios covering: fusion that helps but not
enough to flip the regime, several flips at different ridge points and op
counts, a 5-op fusion that's still not enough, and one **edge case** where
the fused AI lands **exactly** on the ridge point (`ai=4.0, ridge=4.0` —
this must count as compute-bound, since the comparison is `>=`, not `>`).
The candidate's full stdout is compared byte-for-byte (`exact_match =
1.0`) against the reference's. Using `unfused_bytes = 2*n*elem_bytes`
(forgetting the `num_ops` factor that separate passes actually pay for),
or using a strict `>` for the bound check, both reproduce a few rows by
coincidence but mismatch the edge-case row and several of the flip rows.
