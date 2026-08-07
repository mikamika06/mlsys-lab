import sys
sys.path.insert(0, ".")
from attnlab.memory import estimate_peak_memory
from attnlab.traffic import estimate_traffic
from attnlab.masking import validate_mask_dispatch


def test_efficient_uses_less_memory_than_math():
    b, h, s, d = 2, 8, 2048, 64
    m_mem = estimate_peak_memory(b, h, s, d, backend="math")
    e_mem = estimate_peak_memory(b, h, s, d, backend="efficient")
    assert e_mem < m_mem, f"efficient backend memory {e_mem} not less than math {m_mem}"


def test_flash_saves_bandwidth():
    b, h, s, d = 2, 8, 2048, 64
    m_traf = estimate_traffic(b, h, s, d, mode="math")
    f_traf = estimate_traffic(b, h, s, d, mode="flash")
    assert f_traf < m_traf, f"flash traffic {f_traf} not less than math {m_traf}"


def test_mask_conflict_detection():
    res = validate_mask_dispatch(is_causal=True, mask_tensor_present=True)
    assert res == "conflict", f"expected conflict, got {res}"


def test_causal_dispatch():
    res = validate_mask_dispatch(is_causal=True, mask_tensor_present=False)
    assert res == "flash_causal", f"expected flash_causal, got {res}"
