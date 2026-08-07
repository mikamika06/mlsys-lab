import sys

sys.path.insert(0, ".")
from ggufparse.adapter import validate_adapter


def test_adapter_matching():
    base = {"hidden_size": 4096, "num_attention_heads": 32}
    adapter = {"hidden_size": 4096, "num_attention_heads": 32}
    assert validate_adapter(base, adapter) is True


def test_adapter_mismatch():
    base = {"hidden_size": 4096, "num_attention_heads": 32}
    adapter = {"hidden_size": 2048, "num_attention_heads": 16}
    assert validate_adapter(base, adapter) is False
