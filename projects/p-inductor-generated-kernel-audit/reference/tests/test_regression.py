import sys
sys.path.insert(0, ".")
from audit.core import extract_kernel_code, inspect_fusion, analyze_fusion_gap, apply_compilation_controls, optimize_both_sizes
from audit.config import get_default_config, setup_cache_dir


def test_extraction_not_empty():
    code = extract_kernel_code(None, None)
    assert len(code) > 0


def test_inspection_keys():
    res = inspect_fusion("dummy")
    assert "fused" in res


def test_gap_analysis():
    res = analyze_fusion_gap("dummy", "small")
    assert not res["fused"]


def test_controls():
    cfg = apply_compilation_controls({"a": 1})
    assert cfg["a"] == 1


def test_config_defaults():
    cfg = get_default_config()
    assert cfg["cache_enabled"] is True
