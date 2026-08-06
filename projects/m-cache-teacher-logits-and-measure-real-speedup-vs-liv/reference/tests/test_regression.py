import numpy as np
from distillcache.engine import TeacherLogitCache, MockModel, build_teacher_cache, run_distillation_epoch


def test_offline_does_not_call_teacher():
    """Ensure offline distillation mode does not invoke teacher model forward passes."""
    dataset = [
        {"id": 0, "input": np.array([1.0, 2.0])},
        {"id": 1, "input": np.array([3.0, 4.0])}
    ]

    def t_fn(x):
        return x * 2.0

    def s_fn(x):
        return x * 1.5

    teacher = MockModel(t_fn, name="teacher")
    student = MockModel(s_fn, name="student")

    cache = build_teacher_cache(teacher, dataset)
    initial_teacher_calls = teacher.call_count

    stats = run_distillation_epoch(student, dataset, cache=cache, teacher_model=teacher, mode="offline")

    if teacher.call_count != initial_teacher_calls:
        raise AssertionError(f"Teacher model was called {teacher.call_count - initial_teacher_calls} times in offline mode")

    if stats["teacher_calls"] != initial_teacher_calls:
        raise AssertionError("Stats reported non-zero teacher calls during offline distillation")
