import sys
sys.path.insert(0, ".")
from unfoldfix.escape import choose_escape_hatch


def test_all_ten_conversion_failures_covered():
    failures = [
        "UNSUPPORTED_OP_UNFOLD",
        "DYNAMIC_SHAPE_MISMATCH",
        "QUANTIZATION_SCALE_OVERFLOW",
        "CUSTOM_KERNEL_NOT_FOUND",
        "ATTENTION_MASK_RANK_MISMATCH",
        "RMSNORM_AXIS_OUT_OF_BOUNDS",
        "KV_CACHE_STRIDE_INVALID",
        "SILU_FUSION_UNSUPPORTED",
        "EMBEDDING_TABLE_TOO_LARGE",
        "ROPE_FREQ_BASE_INVALID"
    ]
    for f in failures:
        hatch = choose_escape_hatch(f)
        assert hatch != "generic_fallback", f"failure {f} fell back to generic handler"
        assert isinstance(hatch, str) and len(hatch) > 0


def test_unknown_failure_returns_fallback():
    assert choose_escape_hatch("UNKNOWN_FAILURE_CODE") == "generic_fallback"
