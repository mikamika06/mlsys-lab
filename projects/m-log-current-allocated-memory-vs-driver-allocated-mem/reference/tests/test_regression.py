import sys

sys.path.insert(0, ".")
from mps_mem.mock_device import MPSDevice
from mps_mem.fragmentation import fix_oom


def test_fix_prevents_oom():
    dev = MPSDevice(total_mem=2000)
    fix_oom(dev)
