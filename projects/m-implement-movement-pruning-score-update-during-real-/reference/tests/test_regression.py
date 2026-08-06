import numpy as np
from movement.prune import compute_mask_overlap, update_movement_scores


def test_mask_overlap_bounds():
    m1 = np.array([True, False, True, True])
    m2 = np.array([True, True, False, True])
    overlap = compute_mask_overlap(m1, m2)
    assert 0.0 <= overlap <= 1.0


def test_score_update_shape():
    scores = np.zeros((4, 4))
    w = np.ones((4, 4))
    g = np.ones((4, 4))
    res = update_movement_scores(scores, w, g, 0.01)
    assert res.shape == (4, 4)
