import time
import numpy as np


def measure_sync_vs_nosync(mock_gpu_work, total_iterations, sync_every_iter):
    nosync_times = []
    sync_times = []

    for i in range(total_iterations):
        t0 = time.perf_counter()
        mock_gpu_work(enqueue_only=True)
        t1 = time.perf_counter()
        nosync_times.append((t1 - t0) * 1000.0)

    for i in range(total_iterations):
        t0 = time.perf_counter()
        mock_gpu_work(enqueue_only=False)
        t1 = time.perf_counter()
        sync_times.append((t1 - t0) * 1000.0)

    nosync_mean = float(np.mean(nosync_times))
    sync_mean = float(np.mean(sync_times))
    underestimation_pct = float(
        ((sync_mean - nosync_mean) / sync_mean) * 100.0 if sync_mean > 0 else 0.0
    )

    return {
        "nosync_mean_ms": nosync_mean,
        "sync_mean_ms": sync_mean,
        "underestimation_pct": underestimation_pct,
    }
