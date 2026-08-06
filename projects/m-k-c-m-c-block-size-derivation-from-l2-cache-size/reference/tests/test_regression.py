"""Learner regression tests."""

from blocking.cache import derive_l2_blocking
from blocking.registers import derive_register_tile


def test_register_tile_constraints():
    m_r, n_r = derive_register_tile(16, 64, 4)
    assert m_r * (n_r // 16) <= 14
    assert n_r % 16 == 0
    assert m_r > 0 and n_r > 0


def test_cache_blocking_footprint():
    l2_size = 512 * 1024
    elem_size = 4
    alpha = 0.75
    m_r, n_r = derive_register_tile(32, 64, elem_size)
    m_c, k_c = derive_l2_blocking(l2_size, m_r, n_r, elem_size, alpha)

    bytes_used = (k_c * n_r + m_c * k_c) * elem_size
    assert bytes_used <= l2_size * alpha
    assert m_c % m_r == 0
    assert k_c > 0
