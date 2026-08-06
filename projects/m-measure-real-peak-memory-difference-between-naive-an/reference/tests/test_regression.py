import sys
sys.path.insert(0, ".")
from fusedmem.parser import parse_logs

def test_parse_logs_valid():
    u = "Unsloth vram: 4500MB speed: 12.5it/s"
    h = "HF Trainer vram: 7200MB speed: 8.1it/s"
    res = parse_logs(u, h)
    assert res["unsloth_vram"] == 4500.0
    assert res["hf_vram"] == 7200.0
    assert res["vram_saved_pct"] > 0.0

def test_parse_logs_speeds():
    u = "Unsloth vram: 5000MB speed: 10.0it/s"
    h = "HF Trainer vram: 8000MB speed: 7.0it/s"
    res = parse_logs(u, h)
    assert res["unsloth_speed"] == 10.0
    assert res["hf_speed"] == 7.0
