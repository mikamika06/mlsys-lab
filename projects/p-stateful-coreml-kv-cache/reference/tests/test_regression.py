import sys
sys.path.insert(0, ".")
from edge_model.runtime import StatefulRunner
from edge_model.state import define_state_contract

def test_runner_state_reset():
    contract = define_state_contract(2, 4, 16, 128)
    runner = StatefulRunner(contract)
    logits1, l1 = runner.step(10)
    assert l1 == 1
    runner.reset()
    logits2, l2 = runner.step(10)
    assert l2 == 1

def test_sequence_length_increment():
    contract = define_state_contract(1, 2, 8, 64)
    runner = StatefulRunner(contract)
    _, l1 = runner.step(5)
    _, l2 = runner.step(6)
    assert l2 == 2
