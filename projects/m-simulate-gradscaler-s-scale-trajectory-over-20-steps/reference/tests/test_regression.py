import sys

sys.path.insert(0, ".")
from gradscaler.sim import simulate_trajectory, identify_skipped_steps, next_doubling_step


def test_trajectory_length():
    scales = simulate_trajectory([False] * 20)
    assert len(scales) == 20, "Should yield exactly one scale per step."


def test_overflow_resets_successes():
    seq = [False, True, False, False]
    scales = simulate_trajectory(seq, init_scale=100.0, growth_interval=2)
    assert scales == [100.0, 100.0, 50.0, 50.0], f"Expected success reset after overflow, got {scales}"


def test_identify_skipped_steps():
    seq = [False, True, False, True, False]
    assert identify_skipped_steps(seq) == [1, 3], "Skipped steps do not match."


def test_next_doubling_step_calculation():
    assert next_doubling_step([False, False], growth_interval=5) == 4, "Doubling step prediction is off."
