import math


def compare_fsdp1_fsdp2_chunking(global_shape: tuple[int, ...], mesh_size: int, dtype_bytes: int = 4) -> dict:
    """Compare FSDP1 padded sharding vs FSDP2 uneven unpadded DTensor chunking."""
    dim0 = global_shape[0]
    trailing_numel = 1
    for s in global_shape[1:]:
        trailing_numel *= s

    fsdp1_local_dim0 = math.ceil(dim0 / mesh_size)
    fsdp1_padded_dim0 = fsdp1_local_dim0 * mesh_size

    fsdp1_padded_shape = (fsdp1_padded_dim0,) + tuple(global_shape[1:])
    fsdp1_local_shape = (fsdp1_local_dim0,) + tuple(global_shape[1:])

    fsdp1_total_numel = fsdp1_padded_dim0 * trailing_numel
    numel_global = dim0 * trailing_numel

    fsdp1_total_bytes = fsdp1_total_numel * dtype_bytes
    fsdp1_wasted_bytes = (fsdp1_total_numel - numel_global) * dtype_bytes

    base = dim0 // mesh_size
    rem = dim0 % mesh_size
    fsdp2_local_shapes = []
    for r in range(mesh_size):
        ld0 = base + 1 if r < rem else base
        fsdp2_local_shapes.append((ld0,) + tuple(global_shape[1:]))

    fsdp2_total_bytes = numel_global * dtype_bytes
    fsdp2_wasted_bytes = 0

    return {
        "fsdp1_padded_shape": tuple(fsdp1_padded_shape),
        "fsdp1_local_shape": tuple(fsdp1_local_shape),
        "fsdp1_total_bytes": fsdp1_total_bytes,
        "fsdp1_wasted_bytes": fsdp1_wasted_bytes,
        "fsdp2_local_shapes": fsdp2_local_shapes,
        "fsdp2_total_bytes": fsdp2_total_bytes,
        "fsdp2_wasted_bytes": fsdp2_wasted_bytes,
    }
