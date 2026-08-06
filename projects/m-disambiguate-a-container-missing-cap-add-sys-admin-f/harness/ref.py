TEST_CASES = [
    {
        "id": "t1",
        "log": "==ERROR== Profiling failed: perf_event_open system call returned Permission denied. Check kernel settings.",
        "env": {"in_container": True, "has_sys_admin": False, "perf_event_paranoid": 2},
        "expected": "CONTAINER_MISSING_SYS_ADMIN"
    },
    {
        "id": "t2",
        "log": "==ERROR== Failed to initialize sampling: perf_event_open returned Permission denied.",
        "env": {"in_container": False, "has_sys_admin": True, "perf_event_paranoid": 3},
        "expected": "BARE_METAL_PERF_PARANOID"
    },
    {
        "id": "t3",
        "log": "==ERROR== NVML initialization failed: Driver not loaded or version mismatch.",
        "env": {"in_container": False, "has_sys_admin": True, "perf_event_paranoid": 0},
        "expected": "DRIVER_OR_NVML_FAILURE"
    },
    {
        "id": "t4",
        "log": "==ERROR== No CUDA devices found or GPU is unavailable.",
        "env": {"in_container": True, "has_sys_admin": True, "perf_event_paranoid": 0},
        "expected": "DEVICE_UNAVAILABLE"
    },
    {
        "id": "t5",
        "log": "==ERROR== Required capability SYS_ADMIN is missing for kernel profiling instrumentation.",
        "env": {"in_container": True, "has_sys_admin": False, "perf_event_paranoid": 0},
        "expected": "CONTAINER_MISSING_SYS_ADMIN"
    },
    {
        "id": "t6",
        "log": "==ERROR== General permission denied accessing hardware performance counters.",
        "env": {"in_container": False, "has_sys_admin": True, "perf_event_paranoid": 1},
        "expected": "BARE_METAL_PERMISSION_DENIED"
    }
]
