import numpy as np


class TeacherLogitCache:
    """In-memory cache for storing and retrieving teacher logits per sample ID."""

    def __init__(self):
        raise NotImplementedError

    def store(self, sample_id: int, logits: np.ndarray) -> None:
        raise NotImplementedError

    def get(self, sample_id: int) -> np.ndarray:
        raise NotImplementedError

    def has(self, sample_id: int) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class MockModel:
    """Wrapper around a forward function tracking execution calls."""

    def __init__(self, forward_fn, name: str = "model"):
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset_counter(self) -> None:
        raise NotImplementedError


def build_teacher_cache(teacher_model: MockModel, dataset: list) -> TeacherLogitCache:
    """Precompute teacher logits across a dataset and store in TeacherLogitCache."""
    raise NotImplementedError


def run_distillation_epoch(student_model: MockModel, dataset: list, cache: TeacherLogitCache = None, teacher_model: MockModel = None, mode: str = "offline") -> dict:
    """Run one epoch of distillation training in either offline or online mode."""
    raise NotImplementedError


def profile_distillation_overhead(dataset_size: int, teacher_cost_ms: float, student_cost_ms: float, cache_read_cost_ms: float, num_epochs: int = 5) -> dict:
    """Calculate execution time and speedup metrics comparing online vs offline distillation."""
    raise NotImplementedError
