import numpy as np


def get_test_cases():
    np.random.seed(42)
    cases = []

    t1_ref = np.random.randn(10, 10).astype(np.float32)
    t1_tgt = t1_ref + 1e-7
    cases.append((t1_ref, t1_tgt, {"is_logits": False}, "EXACT_MATCH"))

    t2_ref = np.random.randn(10, 10).astype(np.float32)
    t2_tgt = t2_ref + 1e-3 * np.random.randn(10, 10).astype(np.float32)
    cases.append((t2_ref, t2_tgt, {"is_logits": False}, "BENIGN_DRIFT"))

    t3_ref = np.random.randn(10, 10).astype(np.float32)
    t3_tgt = t3_ref.copy()
    t3_tgt[0, 0] = np.nan
    cases.append((t3_ref, t3_tgt, {"is_logits": False}, "NUMERICAL_OVERFLOW"))

    t4_ref = np.random.randn(5, 5).astype(np.float32)
    t4_tgt = np.random.randn(3, 3).astype(np.float32)
    cases.append((t4_ref, t4_tgt, {}, "SHAPE_MISMATCH"))

    t5_ref = np.random.randn(10, 32).astype(np.float32)
    t5_tgt = t5_ref * 5.0
    cases.append((t5_ref, t5_tgt, {"is_logits": True}, "CATASTROPHIC_DIVERGENCE"))

    return cases
