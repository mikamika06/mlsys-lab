## Context

A CPU doesn't fetch individual bytes from memory — it fetches whole 64-byte cache lines. Walking an array with a stride means every access might pull in a full line even though only a few of its bytes are actually used. **Byte efficiency** captures exactly this waste:

$$\text{byte\_efficiency} = \frac{\text{bytes\_used}}{\text{bytes\_fetched}} = \frac{n \cdot \text{width}}{(\text{distinct lines touched}) \cdot 64}.$$

A stride of `1` element (fully packed) is close to $1.0$ efficient; a stride that lands exactly one element per cache line is close to $\text{width}/64$ efficient — most of every fetched line goes unused.

## Task

Implement:

```cpp
void walk(int n, int stride_elems, int width);
double byte_efficiency(int n, int stride_elems, int width);
```

`walk` must call the harness hook `touch(addr)` (declared in `sol.hpp`) once for every byte address element `i` occupies — `[i * stride_elems * width, i * stride_elems * width + width)` — for `i` in `[0, n)`. `byte_efficiency` must call `reset_touch()`, then `walk(...)`, then compute `bytes_used / bytes_fetched` using `touched_line_count()` (also declared in `sol.hpp`) for the fetched-line count.

## Example

For `n = 1000`, `stride_elems = 1`, `width = 4` (a fully contiguous walk of 4-byte elements): every 64-byte line holds 16 consecutive elements, all of them used, so efficiency is `1.0` (up to boundary rounding). For `stride_elems = 16`, `width = 4`: each element lands in its own fresh line (`16 * 4 = 64`), so only `4` of every fetched `64` bytes are used — efficiency `0.0625`.

## What the gate checks

`main.cpp` calls `byte_efficiency` over six fixed `(n, stride_elems, width)` configurations, spanning fully packed, half-packed, one-per-line, and very sparse strides, and prints each ratio. The candidate's numeric output is compared against the reference's with maximum absolute error (`max_abs_err <= 1e-9`). Forgetting to reset the touched-line set between calls, or computing `bytes_fetched` from the wrong line size, produces a ratio that's off by a consistent factor across every case.
