## Context

Mixed-precision training with Adam keeps, per parameter $\Psi$: 2 bytes
for the fp16 parameter, 2 bytes for the fp16 gradient, and 4+4+4 = 12
bytes for the fp32 master weight plus the two Adam moments (momentum,
variance) — 16 bytes total per parameter. ZeRO shards these states
across $N$ data-parallel devices in increasingly aggressive stages:

$$
\begin{aligned}
\text{stage 0 (baseline)}\ \ &: 16\Psi \\
\text{stage 1 } (P_{os})\ \ &: 4\Psi + \dfrac{12\Psi}{N} \\
\text{stage 2 } (P_{os+g})\ \ &: 2\Psi + \dfrac{14\Psi}{N} \\
\text{stage 3 } (P_{os+g+p})\ \ &: \dfrac{16\Psi}{N}
\end{aligned}
$$

Stage 1 shards only the fp32 optimizer states; stage 2 additionally
shards the fp16 gradients; stage 3 additionally shards the fp16
parameters themselves too. Each stage requires strictly more
communication than the last, so in practice you want the **smallest**
stage number whose per-device memory actually fits your budget — no
need to pay stage 3's extra all-gather traffic if stage 1 already fits.

## Task

Implement `min_zero_stage(psi, n_devices, budget_bytes)`:

```python
def min_zero_stage(psi: float, n_devices: int, budget_bytes: float) -> int:
    ...
```

- `psi`: number of model parameters $\Psi$.
- `n_devices`: number of data-parallel devices $N$.
- `budget_bytes`: per-device memory budget, in bytes.

Return the smallest stage in `{0, 1, 2, 3}` whose formula above
evaluates to at most `budget_bytes`. If even stage 3 doesn't fit,
return `-1`.

## Example

```python
min_zero_stage(psi=1e9, n_devices=8, budget_bytes=16e9)
# stage 0 needs 16e9 bytes exactly -> fits -> 0

min_zero_stage(psi=1e9, n_devices=8, budget_bytes=5e9)
# stage 0: 16e9 (no)   stage 1: 4e9+1.5e9=5.5e9 (no)
# stage 2: 2e9+1.75e9=3.75e9 (yes) -> 2

min_zero_stage(psi=1e9, n_devices=8, budget_bytes=1.0)
# even stage 3 (16e9/8 = 2e9) doesn't fit -> -1
```

## What the gate checks

The grader builds 16 cases — 14 randomly generated (`np.random.default_rng`
seeded, `psi` from ~$10^6$ to ~$2\times10^{10}$, `n_devices` from 1 to
64, budgets chosen to land in every stage's fitting range including
"none fit") plus a huge-budget and a near-zero-budget fixed case — and
compares your return value to the same four formulas evaluated directly
in `check.py`. `exact_match` requires the exact integer stage (or `-1`)
on every case. Checking the stages out of order, or returning the
*most* aggressive fitting stage instead of the *least*, will disagree
with the oracle whenever more than one stage would fit.
