## Context

A **butterfly (XOR-shuffle) reduction** has every lane exchange with a
*different* partner at each step, so all 32 lanes end up holding the
same final reduced value (not just lane 0, unlike `shfl_down`'s tree).
`__shfl_xor_sync(mask_bits, val, mask)` reads the value held by lane
`lane XOR mask`.

This CUDA-C subset has no bitwise operators — no `^`, `&`, `|`. For a
power-of-two `mask` (the only kind a butterfly reduction ever uses:
`1, 2, 4, 8, 16`), XOR-ing it into `lane` can still be computed with
plain arithmetic: `mask` corresponds to exactly one bit position.
`(lane / mask) % 2` isolates that bit's value in `lane` (integer
division by the power of two shifts it down to the ones place, `% 2`
reads it). If that bit is `0`, XOR-ing `mask` in turns it into `1` —
which numerically means *adding* `mask` to `lane`. If it's `1`, XOR-ing
`mask` in turns it into `0` — *subtracting* `mask`.

## Task

Implement

```cpp
__global__ void shfl_xor_source_lane(float* out, int mask, int n);
```

For every lane (`lane = threadIdx.x`) in `[0, n)`:

$$
\text{bit} = \left\lfloor \frac{\text{lane}}{\text{mask}} \right\rfloor \bmod 2, \qquad
\text{out}[\text{lane}] = \begin{cases} \text{lane} + \text{mask} & \text{bit} = 0 \\ \text{lane} - \text{mask} & \text{bit} = 1 \end{cases}
$$

## Example

`lane = 5, mask = 4`: `bit = (5/4) % 2 = 1 % 2 = 1`, so
`out[5] = 5 - 4 = 1`. Check against real XOR: `5 = 0b00101`,
`4 = 0b00100`, `5 XOR 4 = 0b00001 = 1` — matches. `lane = 1, mask = 4`:
`bit = (1/4) % 2 = 0`, so `out[1] = 1 + 4 = 5` — and indeed `1 XOR 4 = 5`
(the pairing is always symmetric: if lane `a`'s source is lane `b`, lane
`b`'s source is lane `a`).

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it
once per mask (`1, 2, 4, 8, 16` — all 5 steps of a full 32-lane butterfly
reduction), comparing the 32-entry source-lane table against numpy's own
`np.arange(32) ^ mask` each time. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

across all 5 masks. Using `(lane * mask) % 2` (multiplication instead of
division) or forgetting the `% 2` (isolating the wrong bit range) both
compute a table that's right for a handful of lanes by coincidence but
wrong for most of the other 32.
