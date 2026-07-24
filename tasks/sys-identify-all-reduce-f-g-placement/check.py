import numpy as np


def _oracle_labels(num_blocks: int) -> list:
    """Real oracle, derived independently from Megatron-style tensor
    parallel algebra (not copied from the reference solution's code path):

    Each block is ColumnParallelLinear -> (elementwise) -> RowParallelLinear.
    `f` sits before the column-parallel matmul: identity forward,
    all-reduce backward. `g` sits after the row-parallel matmul:
    all-reduce forward, identity backward.

    A chain of `num_blocks` stacked blocks has `2*num_blocks + 1` edges:
    [in_0, mid_0, out_0==in_1, mid_1, out_1==in_2, ..., mid_{L-1}, out_{L-1}]

    - edge 0 (very first input): only `f` touches it -> bwd all-reduce only.
    - edge 2L (very last output): only `g` touches it -> fwd all-reduce only.
    - odd edges (within a block, column-out to row-in): neither operator
      touches them -> no communication either direction.
    - even interior edges (block i's output fused with block i+1's input):
      `g_i` (fwd all-reduce) and `f_{i+1}` (bwd all-reduce) both sit on the
      same physical edge -> both directions need all-reduce.
    """
    n_edges = 2 * num_blocks + 1
    fwd = np.zeros(n_edges, dtype=bool)
    bwd = np.zeros(n_edges, dtype=bool)

    bwd[0] = True
    fwd[n_edges - 1] = True
    for i in range(1, num_blocks):
        pos = 2 * i
        fwd[pos] = True
        bwd[pos] = True

    labels = []
    for f, b in zip(fwd, bwd):
        if f and b:
            labels.append("both")
        elif f:
            labels.append("fwd_only")
        elif b:
            labels.append("bwd_only")
        else:
            labels.append("none")
    return labels


def grade(sol, fx) -> dict:
    ok = 1.0
    for num_blocks in [1, 2, 3, 4, 6]:
        expected = _oracle_labels(num_blocks)
        try:
            got = sol.classify_allreduce_edges(num_blocks)
            got = list(got)
        except Exception:
            ok = 0.0
            break

        if len(got) != len(expected):
            ok = 0.0
            break
        if [str(x) for x in got] != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
