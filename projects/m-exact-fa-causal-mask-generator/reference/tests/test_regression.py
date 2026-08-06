import sys
sys.path.insert(0, ".")
from famask.generator import generate_causal_mask
from famask.disagreement import disagreement_map
from famask.decode import decode_causal_mask


def test_decode_mask_shape():
    mask = decode_causal_mask(16, alignment="top-left")
    assert mask.shape == (1, 16)


def test_decode_mask_all_true_causal():
    mask = decode_causal_mask(16, alignment="top-left")
    assert mask.all()


def test_generator_identity_alignment():
    m1 = generate_causal_mask(10, 10, alignment="top-left")
    m2 = generate_causal_mask(10, 10, alignment="bottom-right")
    diff = disagreement_map(10, 10)
    assert not diff.any()
    assert (m1 == m2).all()
