def classify_allreduce_edges(num_blocks: int):
    """Classify every edge of a chain of `num_blocks` stacked Megatron-style
    tensor-parallel MLP blocks (ColumnParallelLinear -> elementwise ->
    RowParallelLinear per block) by whether it needs an all-reduce in the
    forward pass, the backward pass, both, or neither.

    Returns a list of length `2 * num_blocks + 1` of labels in
    {"none", "fwd_only", "bwd_only", "both"}, ordered:
    [in_0, mid_0, out_0==in_1, mid_1, out_1==in_2, ..., mid_{L-1}, out_{L-1}]
    """
    raise NotImplementedError('your code here')
