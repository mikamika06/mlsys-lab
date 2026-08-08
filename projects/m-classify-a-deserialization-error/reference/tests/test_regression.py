import struct
import sys

sys.path.insert(0, ".")
from plan import diagnose_load


def test_hardware_compatibility_requires_ampere():
    # Attempting to load an SM 75 engine on SM 80 with hw_compat enabled should fail
    engine1 = struct.pack("<4sIIII", b"TRT\x00", 8, 75, 1, 1)
    res1 = diagnose_load(engine1, env_trt=8, env_sm=80, env_os=1)
    assert res1["status"] == "ERR_SM_ARCH_UNSUPPORTED", f"Expected ERR_SM_ARCH_UNSUPPORTED, got {res1['status']}"

    # Attempting to load an SM 80 engine on SM 75 with hw_compat enabled should fail
    engine2 = struct.pack("<4sIIII", b"TRT\x00", 8, 80, 1, 1)
    res2 = diagnose_load(engine2, env_trt=8, env_sm=75, env_os=1)
    assert res2["status"] == "ERR_SM_ARCH_UNSUPPORTED", f"Expected ERR_SM_ARCH_UNSUPPORTED, got {res2['status']}"
