import numpy as np
from devmesh.mesh import DeviceMesh2D


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
    mesh = DeviceMesh2D(
        mesh_shape=(num_replicas, sharded_size),
        mesh_dim_names=("dp_replicate", "fsdp"),
        device_ids=device_ids,
    )
    return mesh
