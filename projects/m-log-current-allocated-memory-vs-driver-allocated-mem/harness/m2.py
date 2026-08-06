import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"reproduces_oom": 0.0, "fix_prevents_oom": 0.0, "is_fragmentation": 0.0}
    try:
        from mps_mem.mock_device import MPSDevice, OutOfMemoryError
        from mps_mem.fragmentation import reproduce_oom, fix_oom

        class TracedDevice(MPSDevice):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_driver = 0
                self.max_current = 0
            def allocate(self, size):
                res = super().allocate(size)
                self.max_driver = max(self.max_driver, self.driver_allocated_memory())
                self.max_current = max(self.max_current, self.current_allocated_memory())
                return res

        dev1 = TracedDevice(total_mem=1000)
        try:
            reproduce_oom(dev1)
        except OutOfMemoryError:
            out["reproduces_oom"] = 1.0
            if dev1.max_current < dev1.recommended_max_memory():
                out["is_fragmentation"] = 1.0

        dev2 = TracedDevice(total_mem=1000)
        try:
            fix_oom(dev2)
            if dev2.max_current >= dev1.max_current and dev2.max_current > 0:
                out["fix_prevents_oom"] = 1.0
        except OutOfMemoryError:
            pass
    except Exception as e:
        out["_note"] = f"Error: {e}"
    finally:
        sys.path.pop(0)
    return out
