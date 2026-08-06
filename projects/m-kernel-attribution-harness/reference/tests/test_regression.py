from kernel_attr.fallback import FallbackDiagnostics

def test_fallback_diagnostics():
    diag = FallbackDiagnostics()

    cfg_good = {
        "dtype": "float16",
        "head_dim": 64,
        "is_contiguous": True,
        "alignment": 16
    }
    assert diag.diagnose_fallback(cfg_good) == "FLASH_ATTENTION_ELIGIBLE"

    cfg_strided = {
        "dtype": "float16",
        "head_dim": 64,
        "is_contiguous": False,
        "alignment": 16
    }
    assert diag.diagnose_fallback(cfg_strided) == "NON_CONTIGUOUS_LAYOUT"

    cfg_fp32 = {
        "dtype": "float32",
        "head_dim": 64,
        "is_contiguous": True,
        "alignment": 16
    }
    assert diag.diagnose_fallback(cfg_fp32) == "UNSUPPORTED_DTYPE"

    cfg_dim = {
        "dtype": "float16",
        "head_dim": 80,
        "is_contiguous": True,
        "alignment": 16
    }
    assert diag.diagnose_fallback(cfg_dim) == "INVALID_HEAD_DIM"

    cfg_align = {
        "dtype": "float16",
        "head_dim": 64,
        "is_contiguous": True,
        "alignment": 8
    }
    assert diag.diagnose_fallback(cfg_align) == "MISALIGNED_ADDRESS"
