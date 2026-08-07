import numpy as np


class DeviceMesh2D:
    def __init__(self, mesh_shape, mesh_dim_names, device_ids=None):
        if len(mesh_shape) != 2 or len(mesh_dim_names) != 2:
            raise ValueError("DeviceMesh2D requires exactly 2 dimensions")
        self.mesh_shape = tuple(mesh_shape)
        self.mesh_dim_names = tuple(mesh_dim_names)
        total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        if device_ids is None:
            self.device_ids = np.arange(total_devices).reshape(self.mesh_shape)
        else:
            arr = np.array(device_ids)
            if arr.size != total_devices:
                raise ValueError("Device IDs size does not match mesh shape")
            self.device_ids = arr.reshape(self.mesh_shape)

    def get_rank_coords(self, global_rank):
        pos = np.where(self.device_ids == global_rank)
        if len(pos[0]) == 0:
            raise ValueError(f"Rank {global_rank} not found in mesh")
        return int(pos[0][0]), int(pos[1][0])

    def get_dim_group_ranks(self, global_rank, dim_name):
        if dim_name not in self.mesh_dim_names:
            raise ValueError(f"Unknown dimension {dim_name}")
        r, c = self.get_rank_coords(global_rank)
        if dim_name == self.mesh_dim_names[0]:
            return self.device_ids[:, c].tolist()
        else:
            return self.device_ids[r, :].tolist()

    def is_fast_axis(self, dim_name):
        if dim_name not in self.mesh_dim_names:
            raise ValueError(f"Unknown dimension {dim_name}")
        return dim_name == self.mesh_dim_names[1]

    def to_dict(self):
        return {
            "mesh_shape": list(self.mesh_shape),
            "mesh_dim_names": list(self.mesh_dim_names),
            "device_ids": self.device_ids.tolist(),
        }


def verify_hsdp_mesh(mesh, num_nodes, gpus_per_node):
    d0, d1 = mesh.mesh_shape
    names = mesh.mesh_dim_names
    sharded_dim = names[1]

    if sharded_dim != "fsdp":
        return False

    for r in range(d0):
        group_ranks = mesh.device_ids[r, :].tolist()
        nodes = {rank // gpus_per_node for rank in group_ranks}
        if len(nodes) > 1:
            return False

    return True


def construct_hsdp_mesh(world_size, replica_group_size, gpus_per_node):
    if world_size % replica_group_size != 0:
        raise ValueError("world_size must be divisible by replica_group_size")
    num_replicas = world_size // replica_group_size
    sharded_size = replica_group_size

    if sharded_size > gpus_per_node:
        if sharded_size % gpus_per_node != 0:
            raise ValueError("sharded_size must align with gpus_per_node boundary")

    device_ids = np.arange(world_size).reshape((num_replicas, sharded_size))
    return DeviceMesh2D(
        mesh_shape=(num_replicas, sharded_size),
        mesh_dim_names=("dp_replicate", "fsdp"),
        device_ids=device_ids,
    )


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


TEST_CASES = [
    {
        "shape": (2, 4),
        "names": ("dp", "tp"),
        "ranks": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    {
        "shape": (4, 2),
        "names": ("pp", "dp"),
        "ranks": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    {
        "shape": (2, 8),
        "names": ("replica", "fsdp"),
        "ranks": list(range(16)),
    },
    {
        "shape": (8, 2),
        "names": ("node", "local"),
        "ranks": list(range(16)),
    },
    {
        "shape": (4, 4),
        "names": ("dim0", "dim1"),
        "ranks": list(range(16)),
    },
]
