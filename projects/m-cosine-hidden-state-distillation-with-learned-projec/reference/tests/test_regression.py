import numpy as np
from distill.hidden import LearnedProjectionCosineLoss


def test_cosine_loss_normalized():
    loss_fn = LearnedProjectionCosineLoss(student_dim=16, teacher_dim=32, seed=42)
    s = np.ones((2, 4, 16), dtype=np.float64)
    t = np.ones((2, 4, 32), dtype=np.float64)
    loss = loss_fn.forward(s, t)
    assert np.isclose(loss, 0.0, atol=1e-5) or (0.0 <= loss <= 2.0)
    s_scale = s * 100.0
    loss_scale = loss_fn.forward(s_scale, t)
    assert np.isclose(loss, loss_scale, atol=1e-5)
