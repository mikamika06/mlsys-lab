import sys

sys.path.insert(0, ".")
from tokensweep.pareto import find_pareto_front
from tokensweep.sweep import run_sweep


def test_pareto_strictly_non_dominated():
    results = [
        {"budget": 100, "ttft": 10.0, "itl": 10.0},
        {"budget": 200, "ttft": 12.0, "itl": 8.0},
        {"budget": 300, "ttft": 15.0, "itl": 12.0},
    ]
    pareto = find_pareto_front(results)
    for p in pareto:
        for r in results:
            if r == p:
                continue
            dominated = (r["ttft"] <= p["ttft"] and r["itl"] <= p["itl"] and
                         (r["ttft"] < p["ttft"] or r["itl"] < p["itl"]))
            assert not dominated


def test_sweep_output_keys():
    workload = [{"prompt_len": 512, "output_len": 64}]
    res = run_sweep(workload, [1024])
    assert len(res) == 1
    assert "budget" in res[0]
    assert "ttft" in res[0]
    assert "itl" in res[0]
