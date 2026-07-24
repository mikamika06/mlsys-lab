## Context

GGML's **Q6_K** format (used by `llama.cpp` for one of its 6-bit "k-quant"
weight formats) packs a 256-element super-block into four fields:

* `ql` — 128 `uint8` bytes. Each byte stores the **low 4 bits** of two
  quantized values, one in its low nibble and one in its high nibble.
* `qh` — 64 `uint8` bytes. Each byte stores the **high 2 bits** of four
  quantized values, two bits per value.
* `scales` — 16 signed `int8` sub-block scales, one per 16-element
  sub-block of the super-block.
* `d` — one `float`, the super-block's overall scale.

Combining a value's low 4 bits (from `ql`) with its high 2 bits (from `qh`)
gives a signed 6-bit code in `[-32, 31]` (the raw 6-bit unsigned code minus
32). Multiplying by the sub-block's `scales` entry and the super-block's `d`
gives the reconstructed float.

The 256 outputs are produced in **two 128-element chunks** (`n = 0` and
`n = 128`). Chunk `c` (`c = 0, 1`) uses the byte ranges
`ql[64c : 64c+64]`, `qh[32c : 32c+32]`, `scales[8c : 8c+8]`, and writes to
output positions `128c : 128c+128`. Within a chunk, for
$l = 0, \dots, 31$, let $is = \lfloor l/16 \rfloor \in \{0,1\}$ and (using
the chunk-local `ql`, `qh`, `scales`):

$$
\begin{aligned}
q_1 &= \big((\texttt{ql}[l]\ \&\ \texttt{0xF})\ |\ ((\texttt{qh}[l] \gg 0\ \&\ 3) \ll 4)\big) - 32 \\
q_2 &= \big((\texttt{ql}[l{+}32]\ \&\ \texttt{0xF})\ |\ ((\texttt{qh}[l] \gg 2\ \&\ 3) \ll 4)\big) - 32 \\
q_3 &= \big((\texttt{ql}[l] \gg 4)\ |\ ((\texttt{qh}[l] \gg 4\ \&\ 3) \ll 4)\big) - 32 \\
q_4 &= \big((\texttt{ql}[l{+}32] \gg 4)\ |\ ((\texttt{qh}[l] \gg 6\ \&\ 3) \ll 4)\big) - 32
\end{aligned}
$$

which are written, within the chunk, to local output positions
$l,\ l{+}32,\ l{+}64,\ l{+}96$:

$$
\begin{aligned}
y[l]      &= d \cdot \texttt{scales}[is]\;   \cdot q_1 \\
y[l{+}32] &= d \cdot \texttt{scales}[is{+}2] \cdot q_2 \\
y[l{+}64] &= d \cdot \texttt{scales}[is{+}4] \cdot q_3 \\
y[l{+}96] &= d \cdot \texttt{scales}[is{+}6] \cdot q_4
\end{aligned}
$$

## Task

Implement `dequant_q6_k_superblock`:

```python
def dequant_q6_k_superblock(ql: np.ndarray, qh: np.ndarray, scales: np.ndarray, d: float) -> np.ndarray:
    ...
```

* `ql` — `uint8` array of length $128$.
* `qh` — `uint8` array of length $64$.
* `scales` — `int8` (or any signed-integer) array of length $16$.
* `d` — Python `float` (or 0-d array), the super-block scale.

Return a `float64` NumPy array of length $256$: the dequantized super-block,
computed exactly as described above (two 128-element chunks, each unpacking
32 six-bit codes into 4 output groups of 32 each).

## Example

For chunk 0, `l=0`: `q1` combines `ql[0]`'s low nibble with `qh[0]`'s bits
`1:0`; the result (after subtracting 32) is scaled by `scales[0]` and `d` to
give `y[0]`. `q2` combines `ql[32]`'s low nibble with `qh[0]`'s bits `3:2`,
scaled by `scales[2]` and `d`, giving `y[32]`. And so on for `q3` (`y[64]`,
`scales[4]`) and `q4` (`y[96]`, `scales[6]`).

## What the gate checks

**rel_err** — the grader loads a fixture super-block (`q6k_ql.npy`,
`q6k_qh.npy`, `q6k_scales.npy`, `q6k_d.npy`) plus a couple of independently
generated random super-blocks, reconstructs each with a NumPy port of
GGML's `dequantize_row_q6_K`, and checks the global relative L2 error
between your $256$-length output and the oracle's is at most $10^{-3}$.
Swapping the `qh` bit-shift amounts, mixing up which `scales` index feeds
which quarter, using the wrong chunk offsets, or forgetting the `-32`
bias will all show up as a large deviation.
