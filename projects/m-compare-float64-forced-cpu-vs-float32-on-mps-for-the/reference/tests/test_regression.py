import time
import torch
import mpsbench.bench as b


def test_sync_timing_check():
    calls = []

    def fake_sync():
        calls.append(time.perf_counter())

    orig_sync = getattr(torch.mps, "synchronize", None) if hasattr(torch, "mps") else None

    class MockMPS:
        @staticmethod
        def synchronize():
            fake_sync()

    if not hasattr(torch, "mps"):
        torch.mps = MockMPS
    else:
        torch.mps.synchronize = fake_sync

    try:
        def dummy_fn():
            return torch.tensor([1.0])

        b.time_execution(dummy_fn, "mps")
        assert len(calls) == 2
    finally:
        if orig_sync is None:
            if hasattr(torch, "mps"):
                delattr(torch, "mps")
        else:
            torch.mps.synchronize = orig_sync
