import sys
sys.path.insert(0, ".")
from kvbytes.calc import calc_bytes_per_token
from kvbytes.mla import mla_bytes_per_token


def test_calc_positive():
    cfg = {"num_layers": 32, "attn_type": "gqa", "num_heads": 32, "num_kv_heads": 8, "head_dim": 128}
    assert calc_bytes_per_token(cfg) > 0


def test_mla_less_than_mha():
    cfg_mha = {"num_layers": 61, "attn_type": "mha", "num_heads": 128, "head_dim": 128}
    cfg_mla = {"num_layers": 61, "attn_type": "mla", "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_heads": 128}
    assert mla_bytes_per_token(cfg_mla) < calc_bytes_per_token(cfg_mha)
