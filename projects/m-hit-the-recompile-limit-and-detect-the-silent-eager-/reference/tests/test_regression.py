import sys
sys.path.insert(0, ".")
from recompile.tracker import count_recompilations
from recompile.detector import EagerFallbackDetector


def test_tracker_counts_unique_recompiles():
    logs = [
        {"is_recompile": True, "guard_id": 1},
        {"is_recompile": True, "guard_id": 2},
        {"is_recompile": True, "guard_id": 1}
    ]
    assert count_recompilations(logs) == 2


def test_detector_triggers_on_uncompiled():
    det = EagerFallbackDetector(limit=5)
    assert not det.step(True)
    assert det.step(False)


def test_detector_triggers_on_limit_exceeded():
    det = EagerFallbackDetector(limit=2)
    assert not det.step(True)
    assert not det.step(True)
    assert det.step(True)
