import numpy as np
import ref


def check(workdir):
    from distillcache.engine import (
        TeacherLogitCache,
        MockModel,
        build_teacher_cache,
        run_distillation_epoch,
        profile_distillation_overhead,
    )

    out = {
        "cache_correct": 0.0,
        "throughput_ratio": 0.0,
        "saved_teacher_calls": 0.0,
    }

    dataset = ref.make_dummy_dataset(num_samples=10, seq_len=8, seed=42)

    def t_fn(x):
        return x * 3.0 + 1.0

    def s_fn(x):
        return x * 2.0

    teacher = MockModel(t_fn, name="teacher")
    student = MockModel(s_fn, name="student")

    cache = build_teacher_cache(teacher, dataset)

    if len(cache) == len(dataset):
        c_ok = True
        for sample in dataset:
            sid = sample["id"]
            if not cache.has(sid):
                c_ok = False
                break
            got_logits = cache.get(sid)
            want_logits = t_fn(sample["input"])
            if not np.allclose(got_logits, want_logits):
                c_ok = False
                break

        t_calls_after_build = teacher.call_count
        student.reset_counter()

        offline_stats = run_distillation_epoch(
            student, dataset, cache=cache, teacher_model=teacher, mode="offline"
        )

        if (
            c_ok
            and teacher.call_count == t_calls_after_build
            and student.call_count == len(dataset)
            and offline_stats["teacher_calls"] == t_calls_after_build
            and offline_stats["avg_loss"] > 0
        ):
            out["cache_correct"] = 1.0

    prof = profile_distillation_overhead(
        dataset_size=1000,
        teacher_cost_ms=20.0,
        student_cost_ms=5.0,
        cache_read_cost_ms=1.0,
        num_epochs=5,
    )

    out["throughput_ratio"] = float(prof.get("throughput_ratio", 0.0))

    expected_saved = 1000 * (5 - 1)
    if prof.get("teacher_calls_saved") == expected_saved:
        out["saved_teacher_calls"] = 1.0

    return out
