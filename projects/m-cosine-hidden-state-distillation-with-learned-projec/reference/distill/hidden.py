import numpy as np


class LearnedProjectionCosineLoss:
    """Cosine distance loss with a trainable linear projection layer."""

    def __init__(self, student_dim: int, teacher_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(student_dim, teacher_dim) * np.sqrt(2.0 / student_dim)
        self.b = np.zeros((1, teacher_dim))

    def forward(self, student_state: np.ndarray, teacher_state: np.ndarray) -> float:
        proj = student_state @ self.W + self.b
        proj_norm = np.linalg.norm(proj, axis=-1, keepdims=True) + 1e-12
        teacher_norm = np.linalg.norm(teacher_state, axis=-1, keepdims=True) + 1e-12

        p_u = proj / proj_norm
        t_u = teacher_state / teacher_norm

        cosine_sim = np.sum(p_u * t_u, axis=-1)
        loss = np.mean(1.0 - cosine_sim)
        return float(loss)

    def backward(self, student_state: np.ndarray, teacher_state: np.ndarray) -> np.ndarray:
        proj = student_state @ self.W + self.b
        p_norm = np.linalg.norm(proj, axis=-1, keepdims=True) + 1e-12
        t_norm = np.linalg.norm(teacher_state, axis=-1, keepdims=True) + 1e-12

        p_u = proj / p_norm
        t_u = teacher_state / t_norm

        N = student_state.size // student_state.shape[-1]
        dL_dpu = -t_u / N
        dL_dproj = (dL_dpu - p_u * np.sum(dL_dpu * p_u, axis=-1, keepdims=True)) / p_norm
        dL_dstudent = dL_dproj @ self.W.T
        return dL_dstudent
