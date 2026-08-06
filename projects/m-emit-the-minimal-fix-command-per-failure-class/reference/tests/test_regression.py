import sys
sys.path.insert(0, ".")
from buildfix.fix import emit_fix


def test_emit_fix_handles_abi_failure():
    cmd = emit_fix("undefined reference to symbol")
    assert "TORCH_CUDA_ARCH_LIST" in cmd
    assert "--no-build-isolation" in cmd


def test_emit_fix_handles_oom():
    cmd = emit_fix("Killed process out of memory")
    assert "MAX_JOBS" in cmd


def test_emit_fix_default():
    cmd = emit_fix("normal output")
    assert "pip install" in cmd
