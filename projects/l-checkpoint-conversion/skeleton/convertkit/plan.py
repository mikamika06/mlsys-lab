def conversion_plan(tensors, experts=0, out_dtype="F16"):
    """What converting this checkpoint would read, write and cost.

    out_dtype, tensors, unmapped, fanned_out, target_tensors, read_bytes,
    write_bytes, expansion, dequantised_tensors, and a per-tensor list with the
    target name, target shape, byte counts and whether it has to be
    dequantised on the way.
    """
    raise NotImplementedError


def shard_plan(plan, shard_bytes):
    """Split the output into shards of at most shard_bytes, in order.

    Each shard carries its tensor names, its byte total and a
    model-00001-of-000NN.safetensors filename. A tensor larger than the limit
    still has to go somewhere.
    """
    raise NotImplementedError
