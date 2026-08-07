import sys
sys.path.insert(0, ".")

from jaxserv.donation import verify_donation
from jaxserv.bench import compute_breakeven_requests

class MockBuffer:
    def __init__(self):
        self._deleted = False

    def is_deleted(self):
        return self._deleted


def test_donation_invalidation():
    buf = MockBuffer()
    def update_fn(b):
        b._deleted = True
        return "updated"
    res, invalidated = verify_donation(update_fn, buf)
    assert res == "updated"
    assert invalidated is True


def test_breakeven_monotonicity():
    reqs_fast = compute_breakeven_requests(1000.0, 10.0, 2.0)
    reqs_slow = compute_breakeven_requests(1000.0, 10.0, 5.0)
    assert reqs_fast > 0
    assert reqs_slow > reqs_fast
    invalid = compute_breakeven_requests(1000.0, 5.0, 10.0)
    assert invalid == -1
