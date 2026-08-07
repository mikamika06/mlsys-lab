import sys
sys.path.insert(0, ".")
from metatune.tasks import extract_tasks
from metatune.measure import measure_diminishing_returns


def test_extract_tasks_count_and_names():
    spec = ["op1", "op2"]
    tasks = extract_tasks(spec)
    assert len(tasks) == 2
    assert tasks[0]["task_name"] == "f_op1"
    assert tasks[1]["task_name"] == "f_op2"


def test_measure_curve_monotonicity():
    trials = [1, 10, 100]
    latencies = measure_diminishing_returns(trials, 10.0, 2.0)
    assert len(latencies) == len(trials)
    for i in range(len(latencies) - 1):
        assert latencies[i] > latencies[i + 1]
