import pytest
from chunkplan.planner import plan_chunks

def test_planner_basic():
    prompts = [100, 200, 300]
    budget = 250
    res = plan_chunks(prompts, budget)
    assert sum(res) == budget
    assert all(r <= p for r, p in zip(res, prompts))

def test_planner_zero_budget():
    prompts = [100, 200]
    res = plan_chunks(prompts, 0)
    assert res == [0, 0]
