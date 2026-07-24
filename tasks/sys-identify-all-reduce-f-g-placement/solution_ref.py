def classify_allreduce_edges(num_blocks: int):
    """Classify every edge of a chain of `num_blocks` stacked Megatron-style
    tensor-parallel MLP blocks (ColumnParallelLinear -> elementwise ->
    RowParallelLinear per block) by whether it needs an all-reduce in the
    forward pass, the backward pass, both, or neither.

    Returns a list of length `2 * num_blocks + 1` of labels in
    {"none", "fwd_only", "bwd_only", "both"}, ordered:
    [in_0, mid_0, out_0==in_1, mid_1, out_1==in_2, ..., mid_{L-1}, out_{L-1}]

    Rules:
    - `f` (identity fwd / all-reduce bwd) sits immediately before every
      ColumnParallelLinear.
    - `g` (all-reduce fwd / identity bwd) sits immediately after every
      RowParallelLinear.
    - The within-block edge (column-parallel output feeding directly into
      the row-parallel input) needs no communication in either direction.
    - The very first edge only has `f` on it (no preceding `g`), so it
      needs all-reduce on backward only.
    - The very last edge only has `g` on it (no following `f`), so it
      needs all-reduce on forward only.
    - Every interior edge that joins one block's output to the next
      block's input has both a `g` (from the block behind it) and an `f`
      (from the block ahead of it) sitting on it, so it needs all-reduce
      in both directions.
    """
    n_edges = 2 * num_blocks + 1
    needs_fwd = [False] * n_edges
    needs_bwd = [False] * n_edges

    needs_bwd[0] = True                # f at the very start
    needs_fwd[n_edges - 1] = True       # g at the very end

    for i in range(1, num_blocks):
        pos = 2 * i
        needs_fwd[pos] = True
        needs_bwd[pos] = True

    labels = []
    for f, b in zip(needs_fwd, needs_bwd):
        if f and b:
            labels.append("both")
        elif f:
            labels.append("fwd_only")
        elif b:
            labels.append("bwd_only")
        else:
            labels.append("none")
    return labels
