## Context

A cache with $L$-byte lines and $S$ sets maps a byte address $a$ to set

$$\text{set}(a) = \left\lfloor \frac{a}{L} \right\rfloor \bmod S$$

Associativity (the number of ways) never changes which *set* an address
lands in — it only decides how many lines can survive in that set at once
before the oldest gets evicted. So the question "which stride between
consecutive accesses forces them all into the same set?" depends only on
$L$ and $S$.

Walking an array with a stride $S_{\text{stride}}$ that happens to be a
multiple of $L \times S$ is a classic conflict-miss trap: every access maps
to the exact same set, so no matter how large the cache is overall, only
`ways` of those addresses can be resident at once — real code hits this
with power-of-two array dimensions and blindly chosen strides.

## Task

Implement, in your `.cpp` file:

```cpp
long collision_stride(int line_bytes, int sets);
```

Derive the **minimum positive** stride $S_{\text{stride}}$ such that every
address in the sequence $0, S_{\text{stride}}, 2 S_{\text{stride}}, 3
S_{\text{stride}}, \dots$ maps to the same set as address $0$, for the
given `line_bytes` ($L$) and `sets` ($S$).

Stepping by $L$ advances the set index by exactly $1 \bmod S$ per step —
not enough on its own, since after $S-1$ steps it wraps back to $0$ only
once every $S$ steps. Stepping by any multiple of $L$ smaller than $L
\times S$ advances the line index by less than $S$ per step, so its index
mod $S$ cycles through other sets before returning to $0$ — it does not
keep *every* access in the sequence in the same set. The minimum stride
that pins every step to set $0$ is:

$$S_{\text{stride}} = L \times S$$

The driver (`main.cpp`, fixed) runs 7 fixed `(line_bytes, sets)` cases, and
for each one calls your `collision_stride`, then independently checks —
using a ground-truth `set_of(addr, line_bytes, sets)` oracle also in
`main.cpp` that does **not** depend on your answer — whether every address
in $0, S_{\text{stride}}, \dots, 5\,S_{\text{stride}}$ lands in the same
set as address $0$. It prints the stride and that agreement flag for every
case, then the total number of cases that agreed.

## Example

For `line_bytes=64, sets=8`: $S_{\text{stride}} = 64 \times 8 = 512$. Line
indices at $k = 0..5$ are $0, 8, 16, 24, 32, 40$ — every one is $\equiv 0
\pmod 8$, so every address maps to set 0 and the case agrees:

```
case=0 line_bytes=64 sets=8 stride=512 agree=1
```

Across all 7 fixed cases the reference derivation agrees every time:

```
case=6 line_bytes=256 sets=3 stride=768 agree=1
total_agree=7
```

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed number to `exact_match` the same
driver linked against the reference derivation — both the `stride` value
itself for every case AND the `agree` flags. A stride of `0` (the starter)
makes every access literally the same address, so the agreement flags all
trivially read `1` — but the printed `stride` values are `0` instead of
the real collision stride for each case, so the mismatch still shows up
and the gate fails. Only the exact formula $L \times S$, reproduced for
every case, passes.
