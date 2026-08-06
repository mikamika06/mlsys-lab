import sys
sys.path.insert(0, ".")
from affinity.classify import classify_subscription

def test_subscription_sanity():
    assert classify_subscription({"threads": 8, "cores": 8}) == "optimal"
    assert classify_subscription({"threads": 16, "cores": 8}) == "oversubscribed"
    assert classify_subscription({"threads": 4, "cores": 8}) == "under-pinned"
