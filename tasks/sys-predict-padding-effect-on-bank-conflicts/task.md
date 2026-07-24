## Context

Shared memory is split into 32 banks of 4-byte words; a word at index $a$
lives in bank $a \bmod 32$. A warp can service one word per bank per cycle,
so 32 lanes hitting 32 *different* banks costs one wave, while lanes that
collide on the same bank get serialized.

Consider a row-major shared-memory tile with row width (stride) $s$
elements. A common "stencil" access pattern has each lane $t$ of a 32-lane
warp read a different **row** at a fixed column — e.g. gathering a
column of neighbours for a vertical stencil, or transposing a tile. Lane
$t$'s word address is
$$
a_t = t \cdot s, \qquad t = 0, \dots, 31,
$$
so its bank is $b_t = (t \cdot s) \bmod 32$. The conflict degree of the
warp is
$$
c(s) = \max_{k \in \{0,\dots,31\}} \big|\{\, t : b_t = k \,\}\big| .
$$

Multiplication by $s$ modulo $32$ is a **bijection** on $\{0,\dots,31\}$
exactly when $s$ is coprime with $32=2^5$ — i.e. exactly when $s$ is odd.
In that case every lane lands on a distinct bank and $c(s)=1$ (no
conflicts at all, the best possible value, since 32 lanes can never do
better than filling 32 distinct banks once each). If $s$ is even, lanes
collide: e.g. $s$ a multiple of 32 sends *every* lane to bank 0, giving the
worst case $c(s)=32$.

This is why real kernels **pad** a tile's row width $w$ by a small amount
$p$ before using it as the stride: they pick $p$ so that $s = w+p$ is odd.
Note that padding is *not* always needed, and **always adding 1** is not a
safe universal rule either — if $w$ is already odd, adding $1$ makes
$w+1$ even, which can reintroduce conflicts.

## Task

Implement:

```python
def choose_padding(width: int) -> int:
    ...
```

* `width` — a shared-memory tile row width, in elements ($\text{width} \ge
  1$).

Return a non-negative integer `pad` such that using `width + pad` as the
warp's column-access stride $s$ in the model above gives the best possible
conflict degree, $c(s) = 1$ (fully conflict-free) — i.e. `width + pad` must
be odd. Return the *smallest* such `pad` (so: `0` if `width` is already
odd, `1` if `width` is even).

## Example

```python
choose_padding(32)   # 32 is even -> pad = 1; stride 33 is odd -> conflict-free
choose_padding(17)   # 17 is already odd -> pad = 0; stride 17 stays conflict-free
```

`choose_padding(32) == 1` and `choose_padding(17) == 0`. In both cases the
resulting stride is odd, so a warp doing the column-stride access hits 32
distinct banks: `max bank count == 1`.

## What the gate checks

A single gate, **modeled_mem_access**, tries `choose_padding` on a set of
widths (small, large, powers of two, and both parities, plus a few seeded
random values), computes the conflict degree $c(\text{width}+\text{pad})$
under the 32-bank model above for **your** returned `pad`, and separately
brute-forces the true minimum achievable conflict degree over a small
padding search range as an independent lower bound. The score is `1.0`
only if your `pad` is non-negative and its conflict degree matches that
true minimum (which is always `1`, achievable for every `width`) on every
test case; any mismatch or exception makes it `0.0`. A solution that
blindly always pads by `1` fails on odd-width cases, where it turns an
already conflict-free stride into a conflicting even one.
