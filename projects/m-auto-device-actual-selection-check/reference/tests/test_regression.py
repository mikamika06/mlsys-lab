from ovdev.selection import resolve_actual_device


def test_device_selection_accuracy():
    """Verify device selection resolution logic."""
    props = {
        "EXECUTION_DEVICES": ["GPU.0"],
        "AVAILABLE_DEVICES": ["CPU", "GPU.0"]
    }
    dev = resolve_actual_device(props, "THROUGHPUT")
    assert dev == "GPU.0", f"Expected GPU.0, got {dev}"


def test_auto_fallback_detection():
    """Verify fallback detection logic under AUTO mode."""
    props = {
        "AVAILABLE_DEVICES": ["CPU"]
    }
    dev = resolve_actual_device(props, "LATENCY")
    assert dev == "CPU", f"Expected CPU fallback, got {dev}"
