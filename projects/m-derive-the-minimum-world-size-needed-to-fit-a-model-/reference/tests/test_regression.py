import sys
import numpy as np

sys.path.insert(0, ".")
from fsdpfit.derive import derive_world_size
from fsdpfit.shards import simulate_fsdp_shards
from fsdpfit.verify import verify_all_gathered


def test_derive_world_size_bounds():
    ws = derive_world_size(10000, 1000, 3000)
    assert ws >= 4


def test_simulate_fsdp_shards_completeness():
    params = {"w": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    shards = simulate_fsdp_shards(params, world_size=2)
    assert len(shards) == 2
    combined = np.concatenate([shards[0]["w"], shards[1]["w"]])
    assert np.allclose(combined, params["w"])


def test_verify_all_gathered_detects_mismatch():
    params = {"w": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    shards = simulate_fsdp_shards(params, world_size=2)
    assert verify_all_gathered(params, shards) is True
    bad_shards = [{"w": np.array([99.0, 99.0], dtype=np.float32)}, shards[1]]
    assert verify_all_gathered(params, bad_shards) is False
