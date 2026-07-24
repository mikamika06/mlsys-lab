## Context

**False sharing**: two threads each own a variable that lives on the same
cache line. Neither thread ever touches the other's data, but every write
still invalidates the whole line for the other core, forcing a re-fetch —
the threads end up serialized on cache-coherence traffic despite having no
real data dependency.

If `N` thread-local variables are packed back-to-back in an array (thread
`i` owns element `i`), the fix is to pad each variable up to a whole
number of cache lines, so element `i` and element `i+1` never land in the
same line no matter how small the variable itself is.

## Task

Implement

```cpp
PadResult min_padding_for_n_vars(long var_bytes, int line_bytes, long n);
```

returning `{padding_bytes, stride_bytes, total_bytes}` for `n` variables
of `var_bytes` bytes each, padded so every variable occupies a whole
number of `line_bytes`-byte lines:

$$
\text{padding} = \big(\text{line\_bytes} - (\text{var\_bytes} \bmod \text{line\_bytes})\big) \bmod \text{line\_bytes}
$$
$$
\text{stride} = \text{var\_bytes} + \text{padding}, \qquad
\text{total} = \text{stride} \times n
$$

The **outer** `mod line_bytes` matters: when `var_bytes` is already an
exact multiple of `line_bytes` (it already fills whole line(s)), the inner
term `line_bytes - (var_bytes mod line_bytes)` evaluates to `line_bytes`
itself, not `0` — the outer mod is what turns that into the correct
"no padding needed" answer.

## Example

`var_bytes=8, line_bytes=64`: `8 mod 64 = 8`, so
`padding = (64 - 8) mod 64 = 56`, `stride = 8 + 56 = 64` — one whole
64-byte line per variable, as expected for a bare `long` counter.

`var_bytes=64, line_bytes=64` (already exactly one line):
`64 mod 64 = 0`, so `padding = (64 - 0) mod 64 = 0` — correctly zero,
*not* `64`.

## What the gate checks

`main.cpp` runs `min_padding_for_n_vars` over 6 fixed scenarios —
a small counter, a variable that already exactly fills one line, one that
spans two lines with a remainder, a 1-byte flag, one that exactly fills
two lines, and a smaller 32-byte line size — and prints all three result
fields for each. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's. Dropping the outer mod (or
any other formula that happens to work when `var_bytes mod line_bytes`
is nonzero) still fails the `var_bytes=64` and `var_bytes=128` rows, where
the correct answer is exactly zero padding.
