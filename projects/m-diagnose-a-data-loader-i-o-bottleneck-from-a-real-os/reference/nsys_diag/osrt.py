IO_SYSCALLS = {"read", "pread64", "write", "pwrite64", "open", "openat", "close", "fsync", "fdatasync"}
SYNC_SYSCALLS = {"futex", "epoll_wait", "epoll_pwait", "poll", "select", "sem_wait", "pthread_cond_wait"}


def categorize_syscall_time(osrt_rows):
    """Group total time by IO vs Sync syscall categories."""
    io_time = 0.0
    sync_time = 0.0
    other_time = 0.0

    for row in osrt_rows:
        name = row["name"]
        total_time = float(row["total_time_ms"])
        if name in IO_SYSCALLS:
            io_time += total_time
        elif name in SYNC_SYSCALLS:
            sync_time += total_time
        else:
            other_time += total_time

    return {"io_time_ms": io_time, "sync_time_ms": sync_time, "other_time_ms": other_time}


def diagnose_osrt_bottleneck(osrt_rows):
    """Diagnose data-loader I/O bottleneck from OSRT summary rows."""
    cats = categorize_syscall_time(osrt_rows)
    total = cats["io_time_ms"] + cats["sync_time_ms"] + cats["other_time_ms"]
    if total == 0:
        return {
            "primary_bottleneck": "none",
            "io_ratio": 0.0,
            "is_io_bound": False,
            "io_time_ms": 0.0,
            "sync_time_ms": 0.0,
        }

    io_ratio = cats["io_time_ms"] / total
    is_io = io_ratio >= 0.50
    primary = "io_bound" if is_io else ("sync_bound" if cats["sync_time_ms"] > cats["other_time_ms"] else "cpu_bound")

    return {
        "primary_bottleneck": primary,
        "io_ratio": io_ratio,
        "is_io_bound": is_io,
        "io_time_ms": cats["io_time_ms"],
        "sync_time_ms": cats["sync_time_ms"],
    }
