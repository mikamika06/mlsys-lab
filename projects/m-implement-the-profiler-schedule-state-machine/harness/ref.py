SCHEDULE_TEST_CASES = [
    {"skip_first": 2, "wait": 1, "warmup": 1, "active": 2, "repeat": 1, "steps": range(10)},
    {"skip_first": 0, "wait": 0, "warmup": 0, "active": 1, "repeat": 0, "steps": range(5)},
    {"skip_first": 1, "wait": 2, "warmup": 2, "active": 3, "repeat": 2, "steps": range(20)},
    {"skip_first": 5, "wait": 0, "warmup": 2, "active": 1, "repeat": 3, "steps": range(15)},
]

ARITHMETIC_TEST_CASES = [
    {"total_steps": 20, "skip_first": 2, "wait": 1, "warmup": 1, "active": 2, "repeat": 1},
    {"total_steps": 50, "skip_first": 0, "wait": 2, "warmup": 3, "active": 5, "repeat": 0},
    {"total_steps": 15, "skip_first": 5, "wait": 1, "warmup": 1, "active": 2, "repeat": 2},
    {"total_steps": 0, "skip_first": 1, "wait": 1, "warmup": 1, "active": 1, "repeat": 1},
]

DIAGNOSE_TEST_CASES = [
    {
        "config": {
            "activities": ["CPU"],
            "schedule": {"skip_first": 1, "wait": 1, "warmup": 1, "active": 2, "repeat": 1},
            "total_steps": 10,
            "stepped": True,
        },
        "expected": "MISSING_CUDA_ACTIVITY",
    },
    {
        "config": {
            "activities": ["CPU", "CUDA"],
            "schedule": {"skip_first": 1, "wait": 1, "warmup": 1, "active": 2, "repeat": 1},
            "total_steps": 10,
            "stepped": False,
        },
        "expected": "NEVER_STEPPED",
    },
    {
        "config": {
            "activities": ["CPU", "CUDA"],
            "schedule": {"skip_first": 1, "wait": 1, "warmup": 1, "active": 0, "repeat": 1},
            "total_steps": 10,
            "stepped": True,
        },
        "expected": "ZERO_ACTIVE_STEPS",
    },
    {
        "config": {
            "activities": ["CPU", "CUDA"],
            "schedule": {"skip_first": 5, "wait": 5, "warmup": 2, "active": 2, "repeat": 1},
            "total_steps": 8,
            "stepped": True,
        },
        "expected": "TRUNCATED_BEFORE_ACTIVE",
    },
    {
        "config": {
            "activities": ["CPU", "CUDA"],
            "schedule": {"skip_first": 1, "wait": 1, "warmup": 1, "active": 2, "repeat": 1},
            "total_steps": 10,
            "stepped": True,
        },
        "expected": "VALID",
    },
]


def ref_get_step_action(step: int, skip_first: int, wait: int, warmup: int, active: int, repeat: int = 0) -> str:
    """Reference implementation of profiler schedule state machine."""
    if step < 0 or skip_first < 0 or wait < 0 or warmup < 0 or active < 0 or repeat < 0:
        raise ValueError("Step and schedule parameters must be non-negative.")
    if step < skip_first:
        return "NONE"
    s = step - skip_first
    cycle_len = wait + warmup + active
    if cycle_len == 0:
        return "NONE"
    cycle_idx = s // cycle_len
    if repeat > 0 and cycle_idx >= repeat:
        return "NONE"
    offset = s % cycle_len
    if offset < wait:
        return "NONE"
    if offset < wait + warmup:
        return "WARMUP"
    if offset < wait + warmup + active:
        if offset == wait + warmup + active - 1:
            return "RECORD_AND_SAVE"
        return "RECORD"
    return "NONE"


def ref_schedule_summary(total_steps: int, skip_first: int, wait: int, warmup: int, active: int, repeat: int = 0) -> dict:
    """Reference implementation of schedule step breakdown and active ranges."""
    if total_steps < 0:
        raise ValueError("total_steps must be non-negative.")
    none_c = 0
    warmup_c = 0
    record_c = 0
    save_c = 0
    active_ranges = []
    in_active = False
    active_start = 0

    for step in range(total_steps):
        act = ref_get_step_action(step, skip_first, wait, warmup, active, repeat)
        if act == "NONE":
            none_c += 1
            if in_active:
                active_ranges.append((active_start, step))
                in_active = False
        elif act == "WARMUP":
            warmup_c += 1
            if in_active:
                active_ranges.append((active_start, step))
                in_active = False
        elif act == "RECORD":
            record_c += 1
            if not in_active:
                active_start = step
                in_active = True
        elif act == "RECORD_AND_SAVE":
            save_c += 1
            if not in_active:
                active_start = step
                in_active = True
            active_ranges.append((active_start, step + 1))
            in_active = False

    if in_active:
        active_ranges.append((active_start, total_steps))

    return {
        "total_steps": total_steps,
        "none_count": none_c,
        "warmup_count": warmup_c,
        "record_count": record_c,
        "record_and_save_count": save_c,
        "active_ranges": active_ranges,
    }


def ref_diagnose_zero_gpu_events(config: dict) -> str:
    """Reference implementation of zero-GPU event diagnostic."""
    acts = config.get("activities")
    if acts is None or not any(a in acts for a in ("CUDA", "GPU")):
        return "MISSING_CUDA_ACTIVITY"
    if not config.get("stepped", True):
        return "NEVER_STEPPED"
    sched = config.get("schedule", {})
    if sched.get("active", 0) <= 0:
        return "ZERO_ACTIVE_STEPS"
    total_steps = config.get("total_steps", 0)
    skip_first = sched.get("skip_first", 0)
    wait = sched.get("wait", 0)
    if total_steps <= skip_first + wait:
        return "TRUNCATED_BEFORE_ACTIVE"
    return "VALID"
