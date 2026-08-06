import numpy as np
from typing import List, Tuple


def generate_fixtures(seed=42):
    rng = np.random.RandomState(seed)
    student_state = rng.randn(4, 8, 64)
    teacher_state = rng.randn(4, 8, 128)

    student_attn = rng.uniform(0.01, 1.0, size=(4, 12, 16, 16))
    teacher_attn = rng.uniform(0.01, 1.0, size=(4, 12, 16, 16))
    student_attn /= np.sum(student_attn, axis=-1, keepdims=True)
    teacher_attn /= np.sum(teacher_attn, axis=-1, keepdims=True)

    return {
        "student_state": student_state,
        "teacher_state": teacher_state,
        "student_attn": student_attn,
        "teacher_attn": teacher_attn,
    }


def ref_build_tinybert_layer_mapping(student_layers: int, teacher_layers: int) -> List[Tuple[int, int]]:
    stride = teacher_layers // student_layers
    return [(i, (i + 1) * stride) for i in range(student_layers)]


def ref_compute_attention_kl_divergence(student_attn: np.ndarray, teacher_attn: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(teacher_attn, eps, 1.0)
    q = np.clip(student_attn, eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=-1)))
