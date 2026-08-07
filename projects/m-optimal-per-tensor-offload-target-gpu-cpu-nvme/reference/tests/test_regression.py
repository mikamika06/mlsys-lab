import sys

sys.path.insert(0, ".")
from offload_target.placement import select_offload_targets

HARDWARE = {
    0: {"capacity_bytes": 4 * 1024**3, "bandwidth_gbps": 900.0, "latency_us": 1.0},
    1: {"capacity_bytes": 16 * 1024**3, "bandwidth_gbps": 50.0, "latency_us": 10.0},
    2: {"capacity_bytes": 64 * 1024**3, "bandwidth_gbps": 3.2, "latency_us": 100.0},
}

TENSORS = [
    {"id": "t1", "size_bytes": 3 * 1024**3, "access_frequency": 50},
    {"id": "t2", "size_bytes": 3 * 1024**3, "access_frequency": 40},
    {"id": "t3", "size_bytes": 3 * 1024**3, "access_frequency": 30},
]


def test_capacity_constraints_never_violated():
    res = select_offload_targets(TENSORS, HARDWARE)
    usage = res["device_usage"]
    for dev in (0, 1, 2):
        assert (
            usage[dev] <= HARDWARE[dev]["capacity_bytes"]
        ), f"Device {dev} exceeded capacity"


def test_assignments_cover_all_tensors():
    res = select_offload_targets(TENSORS, HARDWARE)
    assignments = res["assignments"]
    assert len(assignments) == len(TENSORS)
    for t in TENSORS:
        assert t["id"] in assignments
