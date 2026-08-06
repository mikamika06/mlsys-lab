from autotune.budget import pick_mode


def test_pick_mode_max_autotune_wins():
    res = pick_mode(10.0, 0.1, 0.05, 0.04, 1000)
    assert res == "max-autotune"
