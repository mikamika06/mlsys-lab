import numpy as np


class LearnedProjectionCosineLoss:
    """Cosine distance loss with a trainable linear projection layer."""

    def __init__(self, student_dim: int, teacher_dim: int, seed: int = 42):
        raise NotImplementedError

    def forward(self, student_state: np.ndarray, teacher_state: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, student_state: np.ndarray, teacher_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError
