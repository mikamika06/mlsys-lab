import pytest
from nsys_diag.nvtx import reconstruct_nvtx_depths


def test_nvtx_stack_validation():
    """Test that mismatched NVTX push/pop sequences are identified."""
    bad_events = [
        {"timestamp_ns": 100, "event_type": "pop", "name": "batch_load"},
    ]
    with pytest.raises(ValueError):
        reconstruct_nvtx_depths(bad_events)

    unmatched_push = [
        {"timestamp_ns": 100, "event_type": "push", "name": "data_fetch"},
    ]
    with pytest.raises(ValueError):
        reconstruct_nvtx_depths(unmatched_push)
