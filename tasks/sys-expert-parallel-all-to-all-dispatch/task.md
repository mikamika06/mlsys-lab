## Context

A Mixture-of-Experts (MoE) layer has $E$ experts but each device only holds
$E / D$ of them (expert parallelism, $D$ = number of devices). A token
$x_i \in \mathbb{R}^d$ is routed to a single expert by a router:

$$
e_i = \arg\max_j (\text{router\_logits})_{ij}, \qquad j \in \{0, \dots, E-1\}.
$$

Because the chosen expert usually does not live on the local device, real
systems perform an **all-to-all** communication step: every device sends
each of its tokens to the device that hosts the token's expert, each device
runs its resident experts on the tokens it received, and a second
**all-to-all** sends the results back to the tokens' original positions.

Experts are placed on devices contiguously by index:

$$
\text{device}(e_i) = \left\lfloor \frac{e_i}{E / D} \right\rfloor .
$$

Crucially, this two-hop communication dance is purely a data-movement
optimization — the *numerical* result for token $i$ must be identical to
simply running it through its assigned expert directly:

$$
y_i = x_i \, W_{e_i}, \qquad W_{e_i} \in \mathbb{R}^{d \times d}.
$$

## Task

Implement `moe_all_to_all_dispatch`:

```python
def moe_all_to_all_dispatch(X: list[list[float]], router_logits: list[list[float]], expert_weight: list[list[list[float]]], num_devices: int):
    ...
```

- `X`: `(N, d)` float64 token embeddings.
- `router_logits`: `(N, E)` float64 router scores. Each token is routed
  top-1 to `expert_id[i] = argmax(router_logits[i])`.
- `expert_weight`: `(E, d, d)` float64, one linear projection matrix per
  expert. `E` is divisible by `num_devices`.
- `num_devices`: number of devices the `E` experts are partitioned across,
  contiguously by index (`device_id = expert_id // (E // num_devices)`).

Simulate the dispatch:

1. Compute `expert_id` and `device_id` for every token from the rule above.
2. **All-to-all dispatch**: for each device, gather the indices/tokens of
   every token routed to it.
3. **Expert compute**: on each device, apply the correct resident expert's
   weight matrix to each of its received tokens: `y = x @ expert_weight[e]`.
4. **All-to-all combine**: scatter every result back into its token's
   original row position, so the output has the same row order as `X`.

Return a tuple `(output, device_counts)`:

- `output`: `(N, d)` float64 array, one row per token, in the *original*
  token order (row `i` is token `i`'s result, regardless of which device
  processed it).
- `device_counts`: length-`num_devices` integer array where
  `device_counts[k]` is the number of tokens dispatched to device `k`.

## Example

```python

X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
router_logits = [[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]]  # -> experts [0, 1, 1]
`[[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]` (which is A, a 2x2 matrix)
#
num_devices = 2  # 1 expert per device

out, counts = moe_all_to_all_dispatch(X, router_logits, expert_weight, num_devices)
# token 0 -> expert 0 (device 0): [1, 0] @ I       = [1, 0]
# token 1 -> expert 1 (device 1): [0, 1] @ 2I       = [0, 2]
# token 2 -> expert 1 (device 1): [1, 1] @ 2I       = [2, 2]
# out     == [[1, 0], [0, 2], [2, 2]]
# counts  == [1, 2]
```

## What the gate checks

The grader builds several seeded `(X, router_logits, expert_weight,
num_devices)` configurations and computes a reference by routing each
token directly to its assigned expert with no dispatch simulation at all
(`y_i = x_i @ expert_weight[expert_id[i]]`), plus the true per-device
token counts from the placement rule.

`max_abs_err` is the worst-case max elementwise absolute difference
between your `output` (restored to original token order) and this
reference, across all configurations (must be `< 1e-5`) — a bug in the
dispatch or combine step scrambles rows or sends tokens to the wrong
expert, which shows up here even though the underlying math is trivial.

`counts_exact_match` is `1.0` only if `device_counts` exactly matches the
true number of tokens per device on every configuration (must equal
`1.0`) — this catches a solution that gets lucky on the output order but
never actually computes a real per-device count.
