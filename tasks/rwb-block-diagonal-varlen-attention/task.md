## Context

Production varlen attention kernels (FlashAttention's `varlen` API,
xformers' `BlockDiagonalMask`) pack many variable-length sequences into
one buffer with **no padding**, and describe the packing with a single
cumulative-length array `cu_seqlens` instead of per-sequence tensors:

$$
\text{cu\_seqlens} = [0, \ell_1, \ell_1+\ell_2, \dots, \textstyle\sum_i \ell_i]
$$

Sequence $i$ occupies rows `cu_seqlens[i] : cu_seqlens[i+1]` of the
packed `q, k, v \in \mathbb{R}^{N\times d}`. Attention is block-diagonal:
row $r$ (belonging to sequence $s(r)$) only attends to other rows in the
same sequence:

$$
\text{score}_{i,j} =
\begin{cases}
\dfrac{q_i \cdot k_j}{\sqrt d} & s(i) = s(j) \\[4pt]
-\infty & s(i) \ne s(j)
\end{cases}
\qquad
O_i = \sum_j \operatorname{softmax}(\text{score}_{i,:})_j \; v_j
$$

This is mathematically identical to unpacking each sequence, running
ordinary dense attention on it independently, and concatenating the
results back together — but real batches are packed and read via
`cu_seqlens`, not pre-split.

## Task

Implement `varlen_block_diagonal_attention`:

```python
def varlen_block_diagonal_attention(q: list[list[float]], k: list[list[float]], v: list[list[float]], cu_seqlens: list[int]) -> list[list[float]]:
    ...
```

- `q`, `k`, `v`: `(N, d)`, packed.
- `cu_seqlens`: `(n_seqs + 1,)` int, `cu_seqlens[0] == 0`,
  `cu_seqlens[-1] == N`. Sequence `i` occupies rows
  `cu_seqlens[i] : cu_seqlens[i+1]`.
- Derive each row's sequence id from `cu_seqlens`, build the
  block-diagonal mask, mask disallowed logits with `-inf` **before**
  softmax, and return the `(N, d)` output.

## Example

```python

# three packed sequences of length 2, 1, 3 -> N = 6
cu_seqlens = [0, 2, 3, 6]
q = [[0.1257302210933933, -0.1321048632913019, 0.6404226504432821, 0.10490011715303971], [-0.535669373161111, 0.36159505490948474, 1.3040000451301372, 0.9470809631292422], [-0.7037352358069926, -1.2654214710460525, -0.6232744625373522, 0.0413259793472436], [-2.3250307746388343, -0.21879166393254573, -1.2459109472530652, -0.7322673547034516], [-0.5442589828573099, -0.31630015636915454, 0.4116305363741328, 1.0425133694426776], [-0.12853466294403426, 1.3664634705496859, -0.6651946734866135, 0.3515100700930197]]
k = [[0.345584192064786, 0.8216181435011584, 0.33043707618338714, -1.303157231604361], [0.9053558666731177, 0.4463745723640113, -0.5369532353602852, 0.5811181041963531], [0.36457239618607573, 0.294132496655526, 0.02842224131579679, 0.5467129866124469], [-0.7364540870016669, -0.16290994799305278, -0.48211931267997826, 0.5988462126346276], [0.03972210748165899, -0.2924567509650886, -0.7819084623568421, -0.2571922406188707], [0.008142180518343508, -0.2756029052993704, 1.2940638143982073, 1.0067243153057943]]
v = [[0.18905338179353307, -0.5227484414807474, -0.41306354339189344, -2.4414673826398556], [1.799707382720902, 1.1441658720372287, -0.32542283686782436, 0.7738065867276614], [0.28121066979764925, -0.5538228364240524, 0.9775674511260357, -0.31055654665915255], [-0.3288239040579627, -0.7921467553588982, 0.45495807124085547, -0.09919805171738795], [0.5452887139646817, -0.6071856998706371, 0.12682784711186987, -0.8922740434297903], [0.8414649723701431, 0.18803508698068597, 0.33057100813532614, 0.41050391297026284]]

out = varlen_block_diagonal_attention(q, k, v, cu_seqlens)
# out[0:2] depends only on rows 0:2 (sequence 0)
# out[2:3] depends only on row 2   (sequence 1, a single token)
# out[3:6] depends only on rows 3:6 (sequence 2)
```

## What the gate checks

The grader loads a committed fixture — a real skewed batch (one
47-token sequence among several short 1-9 token ones, packed together)
— plus several additional seeded synthetic packings, and compares your
output to an oracle that unpacks each segment by `cu_seqlens`, runs
ordinary dense attention independently on each slice in Python, and
concatenates — never calling your function, never hardcoding an
expected value, and structurally unable to leak cross-sequence
information.

`max_abs_err` is the worst per-case max-abs-error across all cases and
must be `<= 1e-5`. Deriving segment ids from the wrong end of
`cu_seqlens`, using `side="left"` where `"right"` is needed at a segment
boundary, or letting any row attend past its segment all produce an
output that diverges from the independently-computed per-sequence
reference.
