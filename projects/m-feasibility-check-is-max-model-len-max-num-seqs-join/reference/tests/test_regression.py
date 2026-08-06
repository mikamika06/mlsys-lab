import sys

sys.path.insert(0, ".")
from feasibility.repair import repair_launch
from feasibility.check import check_feasibility

def test_repair_returns_feasible_params():
    m_len, n_seqs = repair_launch(16384, 512, 16 * 1024**3, 32, 8, 128, 16, 2, 2 * 1024**3)
    feasible = check_feasibility(m_len, n_seqs, 16 * 1024**3, 32, 8, 128, 16, 2, 2 * 1024**3)
    assert feasible, f"repaired parameters ({m_len}, {n_seqs}) are still infeasible"

def test_repair_no_op_when_already_feasible():
    m_len, n_seqs = repair_launch(1024, 16, 64 * 1024**3, 32, 8, 128, 16, 2, 2 * 1024**3)
    assert m_len == 1024
    assert n_seqs == 16
