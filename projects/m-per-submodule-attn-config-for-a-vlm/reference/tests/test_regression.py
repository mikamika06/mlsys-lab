import sys
sys.path.insert(0, ".")
from vlmattn.config import parse_submodule_configs, get_submodule_config
from vlmattn.memory import compute_submodule_bytes, validate_submodule_constraints

CONFIG = {
    "default_attention": {"kv_heads": 4, "head_dim": 64, "window": 1024, "type": "full"},
    "submodules": {
        "vision_encoder": {"kv_heads": 8, "head_dim": 64, "type": "sliding", "window": 512},
        "text_decoder": {"kv_heads": 4, "head_dim": 64, "type": "full"}
    }
}


def test_submodule_parsing():
    parsed = parse_submodule_configs(CONFIG)
    assert "vision_encoder" in parsed
    assert parsed["vision_encoder"]["kv_heads"] == 8
    assert parsed["vision_encoder"]["window"] == 512


def test_memory_computation():
    b_bytes = compute_submodule_bytes(CONFIG, 2, 256)
    assert b_bytes > 0


def test_submodule_constraints():
    assert validate_submodule_constraints(CONFIG) is True
