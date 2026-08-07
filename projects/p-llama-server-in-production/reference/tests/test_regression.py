import sys
sys.path.insert(0, ".")
from server.memory import check_memory_growth
from server.monitor import health_check

def test_memory_growth_stability():
    res = check_memory_growth([10, 20, -30])
    assert res["stable"]

def test_health_check_defaults():
    assert health_check({}) is True
