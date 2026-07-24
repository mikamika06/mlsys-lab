## Context

GPUs have a small, fast on-chip SRAM and a large, slow off-chip HBM. For
attention, the bottleneck is usually not FLOPs but **HBM traffic** — how many
bytes get moved between HBM and SRAM.

A **naive** attention implementation materializes the full $(N,N)$ score
matrix $S=QK^\top$ and probability matrix $P=\operatorname{softmax}(S)$ in
HBM: it writes $S$, reads it back for the softmax, writes $P$, and reads it
back for the $PV$ matmul — on top of reading $Q,K,V$ once and writing the
output $O$ once. That is
$$
\text{elems}_{\text{naive}}(N,d) = \underbrace{3Nd}_{\text{read }Q,K,V} + \underbrace{2N^2}_{\text{write+read }S} + \underbrace{2N^2}_{\text{write+read }P} + \underbrace{Nd}_{\text{write }O} = 4Nd + 4N^2 .
$$

A **tiled** (FlashAttention-style) implementation never writes $S$ or $P$ to
HBM at all — it keeps blocks of $Q,K,V$ resident in SRAM and fuses the whole
computation. Given an SRAM capacity of $M$ elements, it picks a column-tile
size
$$
B_c = \operatorname{clip}\!\left(\left\lceil \frac{M}{4d} \right\rceil,\ 1,\ N\right), \qquad T_c = \left\lceil \frac{N}{B_c} \right\rceil ,
$$
streams $K$ and $V$ from HBM exactly once each over the whole algorithm
($2Nd$ elements), and — because the outer loop is over column tiles — reads
$Q$ and reads+writes the running output accumulator $O$ once per column
tile:
$$
\text{elems}_{\text{tiled}}(N,d,M) = \underbrace{2Nd}_{\text{stream }K,V \text{ once}} + \underbrace{T_c \cdot Nd}_{\text{re-read }Q} + \underbrace{2\,T_c \cdot Nd}_{\text{re-read+write }O} = 2Nd + 3\,T_c\, Nd .
$$

## Task

Implement `hbm_traffic`:

```python
def hbm_traffic(N: int, d: int, M: int, elem_bytes: int = 4) -> dict:
    ...
```

- `N` — sequence length.
- `d` — head dimension.
- `M` — on-chip SRAM capacity, in **elements** (not bytes).
- `elem_bytes` — bytes per element (default `4`, i.e. float32).

Compute `naive_bytes = elems_naive(N, d) * elem_bytes` and
`tiled_bytes = elems_tiled(N, d, M) * elem_bytes` using the formulas above
exactly (including the `ceil` and the clip of $B_c$ into $[1, N]$).

Return
```python
{"naive_bytes": naive_bytes, "tiled_bytes": tiled_bytes, "size_ratio": tiled_bytes / naive_bytes}
```

## Example

```python
hbm_traffic(N=256, d=32, M=20000)
# elems_naive  = 4*256*32 + 4*256**2 = 294912
# Bc = ceil(20000/(4*32)) = 157 (clipped into [1,256])
# Tc = ceil(256/157) = 2
# elems_tiled  = 2*256*32 + 3*2*256*32 = 65536
# size_ratio  ≈ 65536/294912 ≈ 0.2222
```

## What the gate checks

* **rel_err** — the grader computes `size_ratio` from the same formulas on
  several `(N, d, M)` configurations and takes the worst-case relative error
  between your `size_ratio` and the reference's, across all configs
  (must be `<= 1e-9`) — this catches any deviation from the exact formula
  (wrong `ceil`, wrong clip, extra/missing term).
* **size_ratio** — for the specific configuration `N=4096, d=64, M=400000`,
  your returned `size_ratio` must itself be `< 0.1`: for a realistically
  large sequence length and a realistic SRAM budget, tiling should move well
  under a tenth of the HBM traffic that the naive implementation does.
