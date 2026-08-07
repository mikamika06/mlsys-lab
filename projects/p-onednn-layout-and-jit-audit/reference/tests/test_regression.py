import sys
sys.path.insert(0, ".")
from audit.engine import AuditEngine
from audit.model import ModelSequence

def test_engine_logging():
    engine = AuditEngine({})
    assert engine.enable_log() == 1
    events = engine.parse_events()
    assert len(events) >= 5

def test_optimization_reduces_reorders():
    engine = AuditEngine({})
    engine.enable_log()
    assert engine.find_redundant() == 1
    engine.optimize_sequence()
    assert len(engine.get_transitions()) == 0
