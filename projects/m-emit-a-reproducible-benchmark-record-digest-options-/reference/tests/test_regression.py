import sys

sys.path.insert(0, ".")
from benchmark.analysis import cold_start_inflation


def test_cold_start_inflation_positive():
    inf = cold_start_inflation(0.5, 1.0, 1000)
    assert inf > 0, "cold start inclusion must inflate throughput"


def test_inflation_scaling():
    inf1 = cold_start_inflation(0.2, 1.0, 1000)
    inf2 = cold_start_inflation(0.5, 1.0, 1000)
    assert inf1 > inf2, "shorter cold elapsed should show higher inflation"
