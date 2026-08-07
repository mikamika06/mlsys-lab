import sys
import numpy as np

sys.path.insert(0, ".")
from gkdstep.loss import compute_divergence
from gkdstep.drift import measure_sequence_drift


def test_forward_reverse_kl_asymmetry():
    teacher_logits = np.array([[3.0, 0.5, -1.0]])
    student_logits = np.array([[-0.5, 2.0, 0.0]])
    fwd = compute_divergence(teacher_logits, student_logits, divergence_type="forward_kl", temperature=1.0)
    rev = compute_divergence(teacher_logits, student_logits, divergence_type="reverse_kl", temperature=1.0)
    assert fwd > rev + 0.1, f"Expected forward KL ({fwd:.4f}) > reverse KL ({rev:.4f}) + 0.1"


def test_jsd_bounded_and_symmetric():
    t_logits = np.array([[2.0, -1.0]])
    s_logits = np.array([[-1.0, 2.0]])
    jsd1 = compute_divergence(t_logits, s_logits, divergence_type="jsd", temperature=1.0)
    jsd2 = compute_divergence(s_logits, t_logits, divergence_type="jsd", temperature=1.0)
    assert abs(jsd1 - jsd2) < 1e-6
    fwd = compute_divergence(t_logits, s_logits, divergence_type="forward_kl", temperature=1.0)
    assert jsd1 < fwd


def test_on_policy_drift_growth():
    rng = np.random.RandomState(123)
    t_seq = rng.randn(2, 6, 8)
    s_seq = rng.randn(2, 6, 8) + 0.5
    off_res = measure_sequence_drift(t_seq, s_seq, beta=0.0)
    on_res = measure_sequence_drift(t_seq, s_seq, beta=0.8)
    assert on_res["mean_drift"] > off_res["mean_drift"]
