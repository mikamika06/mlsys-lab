import sqlite3


def range_gpu_time(db_path):
    """Calculate overlapping GPU duration for each NVTX range."""
    raise NotImplementedError


def top_kernels_summary(db_path):
    """Return top 3 kernels by total time and their collective percentage share."""
    raise NotImplementedError
