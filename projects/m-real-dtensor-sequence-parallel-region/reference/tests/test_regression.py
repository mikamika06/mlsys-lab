import sys

sys.path.insert(0, ".")
from sp.fix import validate_fix
from sp.memory import measure_memory
from sp.region import build_region


def test_validate_fix_correct_order():
    seq = ["scatter", "compute", "gather"]
    assert validate_fix(seq) is True


def test_validate_fix_incorrect_order():
    seq = ["gather", "compute", "scatter"]
    assert validate_fix(seq) is False


def test_measure_memory_scaling():
    cfg = {"seq_len": 2048, "hidden_size": 1024, "tp_size": 4, "batch_size": 1}
    tp_mem = measure_memory(cfg, "tp_only")
    sp_mem = measure_memory(cfg, "tp_sp")
    assert sp_mem == tp_mem // 4


def test_build_region_properties():
    cfg = {"seq_len": 1024, "hidden_size": 512, "tp_size": 2, "batch_size": 1}
    r = build_region(cfg)
    assert r["sharded_dim"] == 0
    assert r["status"] == "active"
