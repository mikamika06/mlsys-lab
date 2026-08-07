import numpy as np
from compression.pipeline import joint_recipe, find_interaction_flaw

def test_joint_preserves_zeros_count():
    w = np.linspace(10, 20, 100)
    out = joint_recipe(w, 0.5, 8)
    assert np.count_nonzero(out) == 50

def test_joint_is_better_than_pq_on_shifted():
    w = np.linspace(10, 20, 100)
    flaw = find_interaction_flaw(w, 0.5, 8)
    out = joint_recipe(w, 0.5, 8)
    mse_joint = float(np.mean((w - out)**2))
    assert mse_joint < flaw["mse_pq"]

def test_scale_is_not_distorted_by_zeros():
    w = np.array([10.0, 15.0, 20.0, 25.0])
    out = joint_recipe(w, 0.5, 8)
    assert out[2] > 0 and out[3] > 0
