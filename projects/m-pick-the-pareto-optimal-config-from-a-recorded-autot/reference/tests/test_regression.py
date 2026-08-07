import sys

sys.path.insert(0, ".")
from autotune.sweep import load_sweep
from autotune.pareto import compute_pareto
from autotune.select import select_best

SAMPLE_DATA = """0,1.20,32768,64,64,2
1,0.95,49152,128,64,3
2,1.10,32768,64,64,3
3,0.85,98304,128,128,4"""


def test_load_sweep_parses_fields():
    configs = load_sweep(SAMPLE_DATA)
    assert len(configs) == 4
    assert configs[0]["id"] == 0
    assert configs[0]["latency"] == 1.20


def test_compute_pareto_non_empty():
    configs = load_sweep(SAMPLE_DATA)
    frontier = compute_pareto(configs)
    assert len(frontier) > 0, "Pareto frontier cannot be empty on non-empty inputs"


def test_select_best_respects_constraints():
    configs = load_sweep(SAMPLE_DATA)
    best_id = select_best(configs, 40000)
    assert best_id is not None
    matching = [c for c in configs if c["id"] == best_id][0]
    assert matching["shmem"] <= 40000
