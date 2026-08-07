import sys

sys.path.insert(0, ".")
from gptqmem.model import simulate_timeline

def test_hessian_memory_is_fp32():
    timeline = simulate_timeline(4096, 1024, 8, 128)
    hessian_seen = False
    for step in timeline:
        if step["phase"] == "compute_hessian":
            assert step["hessian"] == 4096 * 4096 * 4, f"Hessian was {step['hessian']}"
            hessian_seen = True
    assert hessian_seen
