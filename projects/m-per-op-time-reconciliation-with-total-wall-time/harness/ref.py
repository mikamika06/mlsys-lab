REPORTS = [
    {
        "total_wall_time_us": 10000.0,
        "ops": [
            {"name": "conv1", "real_time_us": 4500.0},
            {"name": "relu1", "real_time_us": 5200.0}
        ]
    },
    {
        "total_wall_time_us": 20000.0,
        "ops": [
            {"name": "matmul", "real_time_us": 8000.0},
            {"name": "add", "real_time_us": 2000.0}
        ]
    },
    {
        "total_wall_time_us": 5000.0,
        "ops": [
            {"name": "norm", "real_time_us": 4900.0}
        ]
    }
]

VALID_DEVICES = ["CPU", "GPU", "NPU", "AUTO", "MULTI"]


def calculate_overhead_ratio(per_op_total, total_wall_time):
    if total_wall_time <= 0:
        return 0.0
    return float((total_wall_time - per_op_total) / total_wall_time)


def reconcile_profile_times(report_data):
    ops = report_data.get("ops", [])
    per_op_total = float(sum(op.get("real_time_us", 0.0) for op in ops))
    total_wall_time = float(report_data.get("total_wall_time_us", 0.0))
    rel_err = calculate_overhead_ratio(per_op_total, total_wall_time)
    return {
        "per_op_total_us": per_op_total,
        "total_wall_time_us": total_wall_time,
        "overhead_ratio": rel_err,
        "reconciled": bool(abs(rel_err) <= 0.05)
    }


def validate_device_target(device_str, available_devices):
    if not isinstance(device_str, str) or not device_str.strip():
        raise RuntimeError("Cannot open empty device name string.")
    target = device_str.split(":")[0].strip()
    if target not in available_devices:
        raise RuntimeError(f"Device '{device_str}' is not in list of supported devices: {available_devices}")
    return True
