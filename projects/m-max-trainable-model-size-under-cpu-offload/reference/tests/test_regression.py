import sys
sys.path.insert(0, ".")
from offload.zero_plus import zero_plus_comm_volume

def test_zero_plus_4x_reduction():
    base_vol = zero_plus_comm_volume(1000000, 2, 16, 4, enable_hpz=False, enable_qgz=False)
    zp_vol = zero_plus_comm_volume(1000000, 2, 16, 4, enable_hpz=True, enable_qgz=True)
    assert base_vol / zp_vol == 4.0, f"Expected 4x reduction, got base {base_vol} vs zero++ {zp_vol}"

def test_zero_plus_single_node_no_hpz_gain():
    base_vol = zero_plus_comm_volume(1000000, 2, 8, 1, enable_hpz=False, enable_qgz=True)
    hpz_vol = zero_plus_comm_volume(1000000, 2, 8, 1, enable_hpz=True, enable_qgz=True)
    assert base_vol == hpz_vol, "hpz should not reduce volume on single node"
