## Context

A multi-head attention activation can be viewed as a tensor with shape
$(B,H,S,D)$: $B$ batch, $H$ heads, $S$ sequence length, $D$ per-head feature
dimension. A query token is identified by the triple $(b,h,s)$, and its
linear id is

$$
\mathrm{id}(b,h,s) = (bH + h)S + s ,
\qquad
N = BHS \text{ tokens total.}
$$

On real SIMT hardware, one lane naturally owns one query token: lane
$\ell = \mathrm{blockIdx.x} \cdot \mathrm{blockDim.x} + \mathrm{threadIdx.x}$
decomposes back into $(b,h,s)$ by undoing the same flattening with integer
division and modulo. Because a launch is warp-rounded (grid $\times$ block
is always a multiple of 32), there are usually a few more lanes than
tokens — those extra lanes must be clearly marked idle, not silently write
garbage or duplicate a real token's slot.

## Task

Implement, as real CUDA-C:

```cpp
__global__ void map_tokens(int* out, int batch, int heads, int seq, int dim, int total_tokens);
```

`lane = blockIdx.x * blockDim.x + threadIdx.x`. If `lane < total_tokens`,
decompose `lane` into `(b, h, s)` (the inverse of the flattening formula
above — `lane` and the token id use the *same* formula, so lane `lane`
owns token `lane`) and store that token id at `out[lane]`. Otherwise store
`-1`.

## Example

For $(B,H,S,D) = (1,2,3,64)$, $N = 6$. With a 32-thread block, lanes
`0..5` store `0,1,2,3,4,5` and lanes `6..31` store `-1`.

## What the gate checks

The driver launches your kernel on 5 different `(batch, heads, seq, dim)`
shapes, each with a warp-aligned launch (`block = 32`,
`grid = ceil(total_tokens / 32)`), and reads back `out[0..grid*block)`. For
every shape, the values must form a **perfect bijection**: every token id
`0..N-1` must appear exactly once, and every lane beyond `N` must store
`-1` — no duplicates, no out-of-range values, no lane silently skipped.
The grader compiles your `.cu` and requires this to hold on all 5 shapes
($\mathrm{exact\_match} = 1.0$). Getting the div/mod decomposition backwards
(e.g. swapping which factor is `heads*seq` vs `seq`) produces a scrambled
but still "valid-looking" set of small integers — it still fails, because
the histogram check catches any token id that's missing or duplicated.
