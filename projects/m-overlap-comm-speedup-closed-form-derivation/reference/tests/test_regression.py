import sys
sys.path.insert(0, ".")

from dsengine.batch_config import resolve_batch_config, validate_batch_config
from dsengine.overlap import compute_speedup, compute_step_time
from dsengine.scaler import DynamicLossScaler


def test_batch_config_requires_data_parallel_size():
    cfg = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 4,
        "data_parallel_size": 4,
    }
    assert validate_batch_config(cfg) is True

    bad_cfg = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 4,
        "data_parallel_size": 4,
    }
    assert validate_batch_config(bad_cfg) is False


def test_resolve_batch_config_with_dp():
    partial = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 4,
        "data_parallel_size": 8,
    }
    resolved = resolve_batch_config(partial)
    assert resolved["train_batch_size"] == 64


def test_overlap_speedup():
    time = compute_step_time(10.0, 10.0, 0.5)
    assert time == 15.0
    speedup = compute_speedup(10.0, 10.0, 0.5)
    assert abs(speedup - 20.0 / 15.0) < 1e-6


def test_loss_scaler_overflow():
    scaler = DynamicLossScaler(init_scale=1024.0, scale_factor=2.0, min_scale=1.0)
    s1 = scaler.update(has_overflow=True)
    assert s1 == 512.0
