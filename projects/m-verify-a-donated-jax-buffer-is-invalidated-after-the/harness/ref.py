import math

TEST_SHAPES = [
    ((1024, 1024), 4, 10),
    ((2048, 4096), 2, 50),
    ((512, 512, 64), 8, 100),
]

BREAKEVEN_CASES = [
    (5000.0, 12.0, 2.0),
    (10000.0, 20.0, 5.0),
    (1000.0, 4.0, 4.5),
    (0.0, 5.0, 1.0),
]

class MockJaxBuffer:
    def __init__(self, data, is_donated=False):
        self.data = list(data)
        self._is_donated = is_donated
        self._deleted = False

    def is_deleted(self):
        return self._deleted


def mock_jit_donate_call(buf, *args):
    if buf._is_donated:
        buf._deleted = True
    return [x + 1 for x in buf.data]


def measure_peak_memory_savings(state_shape, dtype_bytes, num_updates):
    size_bytes = 1
    for dim in state_shape:
        size_bytes *= dim
    size_bytes *= dtype_bytes
    without_donation_peak = size_bytes * 2
    with_donation_peak = size_bytes
    bytes_saved = without_donation_peak - with_donation_peak
    saved_ratio = bytes_saved / without_donation_peak
    return {
        "without_donation_peak": without_donation_peak,
        "with_donation_peak": with_donation_peak,
        "bytes_saved": bytes_saved,
        "saved_ratio": saved_ratio,
    }


def compute_breakeven_requests(compile_time_ms, eager_step_ms, compiled_step_ms) -> int:
    if compiled_step_ms >= eager_step_ms:
        return -1
    diff = eager_step_ms - compiled_step_ms
    return math.ceil(compile_time_ms / diff)
