import sys

sys.path.insert(0, ".")
from leak.detector import MemorySnapshotAnalyzer


def test_leak_does_not_return():
    analyzer = MemorySnapshotAnalyzer({"objects": {"obj1": {"parent": None}}})
    analyzer.fix_retention()
    assert analyzer.simulate_epoch() == 0
