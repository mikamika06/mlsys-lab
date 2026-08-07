import sys
sys.path.insert(0, ".")
from triage.analyzer import parse_startup_logs
from triage.memory import bakeoff_kv_memory
from triage.matrix import extract_support_matrix

def test_parse_startup_logs_detects_features():
    logs = ["warning: flash-attention disabled", "some random line"]
    assert parse_startup_logs(logs) == ["flash-attention"]

def test_bakeoff_kv_memory_allocates_valid_blocks():
    cfg_a = {"layers": 32, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "block_size": 16}
    cfg_b = {"layers": 32, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "block_size": 16}
    res = bakeoff_kv_memory(cfg_a, cfg_b, 1000000)
    assert res["engine_a"]["num_blocks"] == res["engine_b"]["num_blocks"]

def test_extract_support_matrix_parses_classes():
    src = "class Engine:\n    def forward(self):\n        pass\n"
    res = extract_support_matrix(src)
    assert "Engine" in res
    assert "forward" in res["Engine"]
