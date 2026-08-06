from snaptool.footprint import compare_footprint
from snaptool.frames import find_retaining_frame
from snaptool.timeline import build_timeline


def test_retaining_frame_detection():
    mock_snapshot = {
        "device_traces": [[
            {
                "action": "alloc",
                "addr": 0x1000,
                "size": 1024,
                "time": 1,
                "frames": [{"filename": "train.py", "line": 10, "name": "train"}]
            },
            {
                "action": "alloc",
                "addr": 0x2000,
                "size": 4096,
                "time": 2,
                "frames": [{"filename": "utils/logger.py", "line": 50, "name": "log"}]
            }
        ]],
        "model_spec": {
            "param_count": 100,
            "bytes_per_param": 4,
            "optimizer_multiplier": 2.0
        }
    }

    frame, retained_bytes = find_retaining_frame(mock_snapshot)
    assert frame == "utils/logger.py:50:log"
    assert retained_bytes == 4096

    timeline, peak = build_timeline(mock_snapshot)
    assert peak == 5120
    assert len(timeline) == 2

    theo, overhead = compare_footprint(mock_snapshot)
    assert theo == 1200
    assert overhead == 5120 - 1200
