import sys
import numpy as np

sys.path.insert(0, ".")
from safetensors_interop.remap import remap_hf_to_mlx


def test_remap_converts_hf_keys_to_mlx():
    hf_weights = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((4, 4), dtype=np.float16),
        "model.layers.1.self_attn.k_proj.weight": np.ones((4, 4), dtype=np.float16),
        "model.embed_tokens.weight": np.ones((8, 4), dtype=np.float16),
        "model.unknown_layer.weight": np.ones((2, 2), dtype=np.float16),
    }
    rules = {
        "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attention.wq.weight",
        "model.layers.{i}.self_attn.k_proj.weight": "layers.{i}.attention.wk.weight",
        "model.embed_tokens.weight": "embed_tokens.weight",
    }

    remapped, unmapped = remap_hf_to_mlx(hf_weights, rules)

    assert "layers.0.attention.wq.weight" in remapped
    assert "layers.1.attention.wk.weight" in remapped
    assert "embed_tokens.weight" in remapped
    assert "model.layers.0.self_attn.q_proj.weight" not in remapped
    assert "model.unknown_layer.weight" in unmapped
    assert len(unmapped) == 1


def test_untranslated_hf_keys_are_unmapped():
    hf_weights = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((4, 4), dtype=np.float16),
    }
    rules = {}

    remapped, unmapped = remap_hf_to_mlx(hf_weights, rules)

    assert "layers.0.attention.wq.weight" not in remapped
    assert "model.layers.0.self_attn.q_proj.weight" in unmapped
