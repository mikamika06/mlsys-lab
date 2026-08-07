import sys

sys.path.insert(0, ".")
from sp_comm.comm_log import verify_comm_log
from sp_comm.mem_budget import max_seq_len

def test_comm_log():
    log = [{"op": "all_to_all", "volume": 9437184}]
    err = verify_comm_log(log, P=4, S=1024, h=512, b=1, L=12, bytes_per_elem=2)
    assert err < 1e-5

def test_max_seq_len():
    res = max_seq_len(mem_budget=10 * 1024**3, P=4, h=512, b=1, L=12, bytes_per_elem=2)
    assert res["dense"] > 0
    assert res["ulysses"] > res["dense"]
    assert res["ring"] > res["ulysses"]
