import sys

sys.path.insert(0, ".")
from fsdpshards.memory import compute_reshard_memory_profile
from fsdpshards.padding import compare_fsdp1_fsdp2_chunking
from fsdpshards.sharding import get_dtensor_shard_info


def test_reshard_after_forward_saves_memory():
    shapes = [(100, 512), (200, 512), (300, 512)]
    prof_true = compute_reshard_memory_profile(shapes, mesh_size=4, reshard_after_forward=True)
    prof_false = compute_reshard_memory_profile(shapes, mesh_size=4, reshard_after_forward=False)
    assert prof_true["persistent_param_bytes_after_forward"] < prof_false["persistent_param_bytes_after_forward"]
    assert prof_true["saved_bytes_after_forward"] > 0


def test_fsdp2_has_zero_padding_waste():
    res = compare_fsdp1_fsdp2_chunking((101, 256), mesh_size=8)
    assert res["fsdp2_wasted_bytes"] == 0
    assert res["fsdp1_wasted_bytes"] > 0


def test_dtensor_shard_offsets_cover_dimension():
    mesh_size = 4
    global_shape = (10, 32)
    total_dim0 = 0
    for r in range(mesh_size):
        info = get_dtensor_shard_info(global_shape, mesh_size, r, shard_dim=0)
        assert info["offset"][0] == total_dim0
        total_dim0 += info["local_shape"][0]
    assert total_dim0 == global_shape[0]
