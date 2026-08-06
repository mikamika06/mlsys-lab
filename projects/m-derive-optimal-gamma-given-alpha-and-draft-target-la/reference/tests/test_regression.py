from speculative.adaptive import update_gamma


def test_adaptive_increases_on_streak():
    g = 4
    g_new = update_gamma(g, 4)
    assert g_new == 5


def test_adaptive_decreases_on_zero():
    g = 4
    g_new = update_gamma(g, 0)
    assert g_new == 3


def test_adaptive_bounds():
    g = 8
    assert update_gamma(g, 8) <= 8
    g_low = 1
    assert update_gamma(g_low, 0) >= 1
