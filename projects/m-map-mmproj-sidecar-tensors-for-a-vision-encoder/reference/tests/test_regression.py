import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mmproj.mapping import map_tensors

def test_no_duplicate_mappings():
    raw_names = [
        "vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.0.self_attn.v_proj.weight",
        "vision_model.encoder.layers.0.self_attn.out_proj.weight"
    ]
    res = map_tensors(raw_names)
    values = list(res.values())
    assert len(values) == len(set(values)), f"collision detected in mappings: {values}"

def test_embeddings_mapped():
    raw_names = [
        "vision_model.embeddings.patch_embedding.weight",
        "multi_modal_projector.linear_1.weight"
    ]
    res = map_tensors(raw_names)
    assert len(res) == 2, "failed to map basic structure"
