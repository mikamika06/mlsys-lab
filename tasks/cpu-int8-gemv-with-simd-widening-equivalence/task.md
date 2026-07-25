## Context

Quantized inference kernels store weights and activations as `int8` to save
memory bandwidth, but they never accumulate in `int8`. SIMD dot-product
instructions that operate on 8-bit lanes -- NEON's `vmull_s8`/`vmlal_s8`, x86's
`VPMADDWD` -- widen every product, and the running sum they feed, to a 32-bit
lane before adding anything. For a matrix $A \in \mathbb{Z}_8^{m \times n}$ and
vector $x \in \mathbb{Z}_8^n$, the widening GEMV computes

$$
y_i = \sum_{j=0}^{n-1} A_{ij} \, x_j , \quad i = 0,\dots,m-1,
$$

with every product $A_{ij} x_j$ and every partial sum carried in 32 bits. Get
the widening point wrong -- keep the accumulator (or the product) in a
narrower type -- and the sum silently wraps around long before the row is
finished, with no warning at compile time or runtime.

## Task

Implement

```cpp
void gemv_i8(const int8_t* A, const int8_t* x, int32_t* y, int rows, int cols);
```

which computes `y[r] = sum_c A[r*cols+c] * x[c]` for each row `r`. Both the
product `A[r*cols+c] * x[c]` and the accumulator you sum it into must be
32-bit (`int32_t`) at every step -- never `int8_t`, never `int16_t`, not even
as an intermediate you immediately assign away.

## Example

With `cols = 4`, row `[100, 100, 100, 100]` and `x = [100, 100, 100, 100]`:
each product is `100*100 = 10000`, well within `int16_t` range on its own,
but the row sum is `4*10000 = 40000` -- past `int16_t`'s max of `32767`. An
accumulator kept in `int16_t` wraps to `40000 - 65536 = -25536`; the correct
`int32_t` accumulator reports `40000`. Individually-safe products, unsafe
running sum: that gap is exactly what widening the accumulator (not just the
product) is for.

## What the gate checks

The driver (`main.cpp`) fills an 8x32 `int8_t` matrix and a 32-element
`int8_t` vector from a fixed seeded generator (full `int8_t` range, so
products up to `128*128` in magnitude appear, and 32 of them get summed per
row), calls `gemv_i8`, and prints the resulting `y` vector. `verify_native.sh`
compiles `solve.cpp` and `ref.cpp` against the same `main.cpp` with
`clang++ -O2 -std=c++20` and requires

$$
\mathrm{exact\_match} = 1 \iff \text{every printed } y_r \text{ matches the reference}
$$

An implementation that narrows the product or the accumulator to `int16_t`
computes the right answer for short rows or small values, then silently
disagrees with the reference the moment a row's running sum crosses 32767 in
magnitude -- which happens well before this task's 32-element rows are done,
by design.
