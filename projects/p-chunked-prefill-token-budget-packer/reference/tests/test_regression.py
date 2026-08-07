import sys
sys.path.insert(0, ".")
from packer.packer import Packer, Request

def test_no_starvation():
    p = Packer(10)
    r1 = Request("r1", 5)
    p.add_request(r1)

    # Process r1 so it enters decode phase
    p.step()
    assert r1.is_decode, "Request should have transitioned to decode"

    # Add a huge prefill
    r2 = Request("r2", 100)
    p.add_request(r2)

    # Run step; r1 must be scheduled for decode despite the huge prefill
    alloc = p.step()
    assert alloc.get("r1") == 1, "Decode request was starved!"
    assert sum(alloc.values()) <= 10, "Budget exceeded!"

def test_budget_respected():
    p = Packer(10)
    r1 = Request("r1", 100)
    p.add_request(r1)

    alloc = p.step()
    assert sum(alloc.values()) <= 10, "Budget exceeded!"
