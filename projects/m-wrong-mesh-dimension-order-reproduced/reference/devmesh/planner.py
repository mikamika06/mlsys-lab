import numpy as np
from devmesh.mesh import DeviceMesh2D


def assign_fast_axis(dim_names, dim_sizes, bandwidth_matrix, gpus_per_node):

    if len(dim_names) != 2 or len(dim_sizes) != 2:
        raise ValueError("Planner requires exactly 2 dimensions")

    n = bandwidth_matrix.shape[0]
    intra_bw_sum = 0.0
    inter_bw_sum = 0.0
    intra_count = 0
    inter_count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                if (i // gpus_per_node) == (j // gpus_per_node):
                    intra_bw_sum += bandwidth_matrix[i, j]
                    intra_count += 1
                else:
                    inter_bw_sum += bandwidth_matrix[i, j]
                    inter_count += 1

    avg_intra = intra_bw_sum / max(1, intra_count)
    avg_inter = inter_bw_sum / max(1, inter_count)

    d0_size, d1_size = dim_sizes[0], dim_sizes[1]
    d0_name, d1_name = dim_names[0], dim_names[1]

    if avg_intra >= avg_inter:
        ordered_names = (d0_name, d1_name)
        ordered_sizes = (d0_size, d1_size)
    else:
        ordered_names = (d1_name, d0_name)
        ordered_sizes = (d1_size, d0_size)

    device_ids = np.arange(n).reshape(ordered_sizes)
    return DeviceMesh2D(
        mesh_shape=ordered_sizes,
        mesh_dim_names=ordered_names,
        device_ids=device_ids,
    )
