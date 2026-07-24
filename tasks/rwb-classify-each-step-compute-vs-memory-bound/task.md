## Context

In continuous batching, one serving **step** runs a single fused forward
pass over every token due that step: one new token per already-running
sequence (`decode_tokens`) plus however many fresh prompt tokens are
being prefilled (`prefill_tokens`). Whether that step is **compute-bound**
or **memory-bound** is a roofline question.

Take one square weight matrix of a linear layer, shape
$(d_{\text{model}}, d_{\text{model}})$, applied to $T = \text{decode\_tokens} + \text{prefill\_tokens}$
tokens in the step:

$$
\text{FLOPs} = 2\, d_{\text{model}}^2\, T, \qquad
\text{Bytes} = d_{\text{model}}^2 \cdot \text{bytes\_per\_param}
$$

(the weight is loaded once regardless of how many tokens use it, so
`Bytes` doesn't grow with $T$). Their ratio is the step's **arithmetic
intensity**:

$$
\text{AI}(T) = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2\,T}{\text{bytes\_per\_param}}
$$

($d_{\text{model}}$ cancels — the bound only depends on the token count
and the weight's bytes-per-parameter). The hardware has its own **ridge
point**, the arithmetic intensity at which it stops being able to hide
memory latency behind compute:

$$
\text{AI}_{\text{ridge}} = \frac{\text{peak\_flops}}{\text{peak\_bandwidth}}
$$

A step is **compute-bound** if $\text{AI}(T) \ge \text{AI}_{\text{ridge}}$
(there's enough work per byte to keep the FLOPs units busy) and
**memory-bound** otherwise (the step finishes moving bytes before it runs
out of arithmetic to do). This is exactly why decode-only steps (small
$T$, one token per sequence) are memory-bound in production, while large
prefill chunks push $T$ up until the step becomes compute-bound.

## Task

Implement `classify_steps`:

```python
def classify_steps(
    steps: list[tuple[int, int]],
    bytes_per_param: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> list[str]:
    ...
```

- `steps`: a list of `(decode_tokens, prefill_tokens)` pairs, one per
  serving step (both non-negative ints).
- `bytes_per_param`: bytes per weight element (e.g. `2.0` for fp16).
- `peak_flops`: the accelerator's peak FLOPs/second.
- `peak_bandwidth`: the accelerator's peak memory bandwidth, bytes/second.

For each step, compute $T = \text{decode\_tokens} + \text{prefill\_tokens}$,
its arithmetic intensity $\text{AI}(T)$, and the hardware's
$\text{AI}_{\text{ridge}}$ as defined above, and return the label
`"compute"` if $\text{AI}(T) \ge \text{AI}_{\text{ridge}}$, else
`"memory"`. Return a list of labels, same length and order as `steps`.

## Example

```python
classify_steps(
    steps=[(32, 0), (2, 0), (0, 200)],
    bytes_per_param=2.0, peak_flops=200e12, peak_bandwidth=4e12,
)
# AI_ridge = 200e12 / 4e12 = 50
# step 0: T=32,  AI = 2*32/2 = 32  < 50 -> "memory" (pure decode, small batch)
# step 1: T=2,   AI = 2*2/2  = 2   < 50 -> "memory" (decode, even smaller)
# step 2: T=200, AI = 2*200/2 = 200 >= 50 -> "compute" (large prefill chunk)
# -> ["memory", "memory", "compute"]
```

## What the gate checks

The grader builds several `(steps, bytes_per_param, peak_flops,
peak_bandwidth)` scenarios — hand-picked steps sitting exactly on the
compute/memory boundary (`AI(T) == AI_ridge` exactly, which must resolve
to `"compute"`), steps just above and just below it, pure-decode and
pure-prefill steps, and a batch of steps from a seeded NumPy generator —
and computes the reference labels independently in NumPy from the
formulas above, never calling your function or hardcoding an expected
label list.

`exact_match` is the fraction of steps (pooled across every scenario)
whose label matches the oracle's, and the gate requires `1.0`. Using
`decode_tokens` or `prefill_tokens` alone instead of their sum, getting
the boundary comparison direction backwards, or forgetting that
$d_{\text{model}}$ cancels out of the ratio (e.g. reintroducing it into
the comparison) will mislabel at least the boundary case.
