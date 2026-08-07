import sys
sys.path.insert(0, ".")
from profiler_study.modes import ProfilerEnvironment
from profiler_study.analyzer import ProfilerAnalyzer

def test_environment_baseline():
    env = ProfilerEnvironment(100.0)
    assert env.measure("clean") == 100.0

def test_analyzer_selection():
    env = ProfilerEnvironment(100.0)
    analyzer = ProfilerAnalyzer(env)
    assert analyzer.select_mode("macro") == "sampling"

def test_discrepancy_check():
    env = ProfilerEnvironment(100.0)
    analyzer = ProfilerAnalyzer(env)
    assert analyzer.check_discrepancy("sampling", 10.0)
