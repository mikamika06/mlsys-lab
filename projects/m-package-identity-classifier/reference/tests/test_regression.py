import sys

sys.path.insert(0, ".")
from fa_classifier.classifier import classify_package
from fa_classifier.dispatch import resolve_dispatch_target
from fa_classifier.gating import check_hardware_gating


def test_invalid_hardware_override():
    pkg_info = {
        "name": "flash_attn",
        "version": "2.5.0",
        "cxx11_abi": True,
        "cuda_compiled": True,
        "has_fp8": True,
        "has_varlen": True,
    }
    hw_info = {"sm_major": 7, "sm_minor": 5, "has_bf16": False}

    identity = classify_package(pkg_info)["identity"]
    gate = check_hardware_gating(identity, hw_info)
    assert not gate["compatible"], "SM75 should not be compatible with FA2"

    dispatch = resolve_dispatch_target(pkg_info, hw_info)
    assert dispatch["status"] == "REJECTED_HARDWARE"
    assert dispatch["target_kernel"] == "TRITON_FALLBACK"


def test_valid_fa2_dispatch():
    pkg_info = {
        "name": "flash_attn",
        "version": "2.1.0",
        "cxx11_abi": True,
        "cuda_compiled": True,
        "has_fp8": False,
        "has_varlen": True,
    }
    hw_info = {"sm_major": 8, "sm_minor": 0, "has_bf16": True}

    dispatch = resolve_dispatch_target(pkg_info, hw_info)
    assert dispatch["status"] == "ACCEPTED"
    assert dispatch["target_kernel"] == "FA2_CUTLASS_FLASH"
