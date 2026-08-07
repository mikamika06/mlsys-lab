import sys
import numpy as np

sys.path.insert(0, ".")
from devmesh.mesh import DeviceMesh2D
from devmesh.hsdp import construct_hsdp_mesh, verify_hsdp_mesh


def test_mesh_dimension_order():
    mesh = construct_hsdp_mesh(world_size=16, replica_group_size=4, gpus_per_node=8)
    assert mesh.is_fast_axis("fsdp")
    assert not mesh.is_fast_axis("dp_replicate")
    assert verify_hsdp_mesh(mesh, num_nodes=2, gpus_per_node=8)


def test_inverted_mesh_rejected():
    inverted_ids = np.arange(16).reshape((4, 4))
    mesh_inv = DeviceMesh2D(
        mesh_shape=(4, 4),
        mesh_dim_names=("fsdp", "dp_replicate"),
        device_ids=inverted_ids,
    )
    assert not verify_hsdp_mesh(mesh_inv, num_nodes=2, gpus_per_node=8)
