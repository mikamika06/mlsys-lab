## Context

A transformer decode cache stores key and value vectors for every token and
every attention head. For $T$ tokens, $H$ heads, a key/value selector
$k \in \{0,1\}$, head dimension $D$, and element size $E$ bytes, the logical
element is $KV[t,h,k,d]$.

Two access patterns compete for the same physical byte layout:

- **whole-token write**: appending the newest token $t = T-1$ touches every
  $(h, k, d)$ for that one $t$, in order $h$, then $k$, then $d$.
- **per-head decode read**: streaming one head's K/V across every existing
  token touches every $(t, k, d)$ for that one $h$, in order $t$, then $k$,
  then $d$, for every $h$.

The harness models a $64$-byte cache line, and the tested shape satisfies

$$
2DE = 64,
$$

so one head's complete K+V record for one token fits exactly in one cache
line. Token-major layout ("THKD" -- token, then head, then key/value, then
dimension)

$$
\operatorname{addr}(t,h,k,d) = \operatorname{base} + \big(((tH+h)\cdot 2 + k)D + d\big) E
$$

keeps a whole token's multi-head record contiguous (cheap write) while never
splitting one head's one-line K+V record across two lines (cheap per-head
read), which is why it beats the other orderings on total traffic.

## Task

Implement, in `solve.cpp`:

```cpp
long thkd_addr(long base, int T, int H, int D, int E, int t, int h, int k, int d);
```

using exactly the THKD formula above:

```
index = (((t*H + h)*2 + k)*D + d)
addr  = base + index * E
```

`main.cpp` (fixed) calls `thkd_addr` to build the write trace for the newest
token and the read trace for a per-head decode sweep, replays both through a
deterministic 64-byte-line, 32-set, 4-way LRU cache model (`touch_byte`,
declared in `sol.hpp`, defined in `main.cpp`), and does the same for four
other fixed orderings (TKHD, TDHK, HTKD, HKTD) so you can see THKD's traffic
against the alternatives. It prints every layout's write/read/total miss
count and the name of the layout with the lowest total.

## Example

For $T=32, H=8, D=16, E=2$ (so $2DE=64$), the driver prints one line per
layout, e.g.

```
THKD write_misses=<w> read_misses=<r> total=<w+r>
TKHD write_misses=<w> read_misses=<r> total=<w+r>
...
best=THKD
```

## What the gate checks

`exact_match` requires the candidate's full stdout to equal the reference's
stdout byte-for-byte. The reference's `thkd_addr` produces the lowest total
miss count of the five layouts, so `best=THKD` is correct and the whole
printed table matches. A wrong `thkd_addr` (for example one that ignores its
arguments) changes the THKD line and can even change which layout the driver
reports as `best`, so the mismatch is caught immediately -- the obvious wrong
approach (any layout that isn't token-major-then-head-then-kv-then-dim, or no
addressing logic at all) fails because its printed miss counts do not equal
the reference's.
