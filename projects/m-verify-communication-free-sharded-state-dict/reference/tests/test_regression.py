"""Regression tests for FSDP2 fully_shard call sequences."""

from fsdp_verify.reconstruct import reconstruct_fully_shard_sequence

SAMPLE_TREE = {
    "should_shard": True,
    "children": {
        "backbone": {
            "should_shard": True,
            "children": {
                "layer0": {"should_shard": True, "children": {}},
                "layer1": {"should_shard": True, "children": {}},
            },
        },
        "head": {"should_shard": True, "children": {}},
    },
}


def test_reconstruct_is_strict_bottom_up():
    seq = reconstruct_fully_shard_sequence(SAMPLE_TREE)
    assert "backbone.layer0" in seq
    assert "backbone.layer1" in seq
    assert seq.index("backbone.layer0") < seq.index("backbone")
    assert seq.index("backbone.layer1") < seq.index("backbone")


def test_no_parent_before_children():
    seq = reconstruct_fully_shard_sequence(SAMPLE_TREE)
    assert seq.index("backbone") < seq.index("")
    assert seq.index("head") < seq.index("")
