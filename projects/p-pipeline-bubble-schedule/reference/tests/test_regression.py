import sys
sys.path.insert(0, ".")
from sched.pipeline import PipelineScheduler

def test_gpipe_basic():
    ps = PipelineScheduler(4, 8)
    assert ps.gpipe_utilization() > 0.4

def test_1f1b_schedule():
    ps = PipelineScheduler(4, 8)
    assert len(ps.schedule_1f1b()) > 0

def test_interleaved():
    ps = PipelineScheduler(4, 8)
    assert ps.interleaved_memory(2) > 0

def test_zero_bubble():
    ps = PipelineScheduler(4, 8)
    assert ps.zero_bubble_schedule()["valid"] is True

def test_traffic():
    ps = PipelineScheduler(4, 8)
    assert ps.evaluate_traffic([0.9, 0.8]) >= 0.8

def test_budget():
    ps = PipelineScheduler(4, 8)
    assert ps.check_activation_budget(100) is True
