import numpy as np


class TeacherLogitCache:
    """In-memory cache for storing and retrieving teacher logits per sample ID."""

    def __init__(self):
        self._store = {}

    def store(self, sample_id: int, logits: np.ndarray) -> None:
        """Store teacher logits for a given sample ID."""
        self._store[sample_id] = np.array(logits, copy=True)

    def get(self, sample_id: int) -> np.ndarray:
        """Retrieve stored teacher logits for a sample ID."""
        if sample_id not in self._store:
            raise KeyError(f"Sample ID {sample_id} not found in cache")
        return self._store[sample_id]

    def has(self, sample_id: int) -> bool:
        """Check if sample ID is present in cache."""
        return sample_id in self._store

    def clear(self) -> None:
        """Clear all stored entries from cache."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class MockModel:
    """Wrapper around a forward function tracking execution calls."""

    def __init__(self, forward_fn, name: str = "model"):
        self.forward_fn = forward_fn
        self.name = name
        self.call_count = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return self.forward_fn(x)

    def reset_counter(self) -> None:
        self.call_count = 0


def build_teacher_cache(teacher_model: MockModel, dataset: list) -> TeacherLogitCache:
    """Precompute teacher logits across a dataset and store in TeacherLogitCache."""
    cache = TeacherLogitCache()
    for sample in dataset:
        sample_id = sample["id"]
        inp = sample["input"]
        logits = teacher_model(inp)
        cache.store(sample_id, logits)
    return cache


def run_distillation_epoch(student_model: MockModel, dataset: list, cache: TeacherLogitCache = None, teacher_model: MockModel = None, mode: str = "offline") -> dict:
    """Run one epoch of distillation training in either offline or online mode."""
    if mode not in ("offline", "online"):
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "offline" and cache is None:
        raise ValueError("Cache must be provided for offline distillation")

    if mode == "online" and teacher_model is None:
        raise ValueError("Teacher model must be provided for online distillation")

    total_loss = 0.0
    num_samples = len(dataset)

    for sample in dataset:
        sample_id = sample["id"]
        inp = sample["input"]

        if mode == "offline":
            t_logits = cache.get(sample_id)
        else:
            t_logits = teacher_model(inp)

        s_logits = student_model(inp)
        loss = float(np.mean((s_logits - t_logits) ** 2))
        total_loss += loss

    avg_loss = total_loss / max(num_samples, 1)
    return {
        "num_samples": num_samples,
        "avg_loss": avg_loss,
        "student_calls": student_model.call_count,
        "teacher_calls": teacher_model.call_count if teacher_model is not None else 0
    }


def profile_distillation_overhead(dataset_size: int, teacher_cost_ms: float, student_cost_ms: float, cache_read_cost_ms: float, num_epochs: int = 5) -> dict:
    """Calculate execution time and speedup metrics comparing online vs offline distillation."""
    online_per_step = teacher_cost_ms + student_cost_ms
    online_total_ms = dataset_size * online_per_step * num_epochs

    cache_build_ms = dataset_size * teacher_cost_ms
    offline_per_step = cache_read_cost_ms + student_cost_ms
    offline_train_ms = dataset_size * offline_per_step * num_epochs
    offline_total_ms = cache_build_ms + offline_train_ms

    ratio = online_total_ms / max(offline_total_ms, 1e-9)
    train_loop_ratio = online_per_step / max(offline_per_step, 1e-9)
    saved_calls = dataset_size * max(num_epochs - 1, 0)

    return {
        "online_total_ms": float(online_total_ms),
        "offline_total_ms": float(offline_total_ms),
        "cache_build_ms": float(cache_build_ms),
        "throughput_ratio": float(ratio),
        "train_loop_throughput_ratio": float(train_loop_ratio),
        "teacher_calls_saved": int(saved_calls)
    }
