"""Oracle reference data generators and helpers."""

STATE_DICT_CASES = [
    (
        {
            "w1": {
                "shape": [1024, 512],
                "shard_dim": 0,
                "dtype_bytes": 4,
                "placements": {0: {"start": 0, "end": 256}, 1: {"start": 256, "end": 512}},
            },
            "b1": {
                "shape": [1024],
                "shard_dim": 0,
                "dtype_bytes": 4,
                "placements": {0: {"start": 0, "end": 256}, 1: {"start": 256, "end": 512}},
            },
        },
        4,
        0,
    ),
    (
        {
            "w1": {
                "shape": [1024, 512],
                "shard_dim": 0,
                "dtype_bytes": 4,
                "placements": {0: {"start": 0, "end": 512}, 1: {"start": 512, "end": 1024}},
            },
        },
        2,
        0,
    ),
    (
        {
            "w1": {
                "shape": [128, 64],
                "shard_dim": 0,
                "dtype_bytes": 2,
                "placements": {
                    0: {"start": 0, "end": 64, "requires_comm": True},
                    1: {"start": 64, "end": 128},
                },
            },
        },
        2,
        0,
    ),
    (
        {
            "w1": {
                "shape": [256, 256],
                "shard_dim": 0,
                "dtype_bytes": 4,
                "placements": {0: {"start": 0, "end": 128}, 1: {"start": 128, "end": 256}},
            },
            "w2": {
                "shape": [512, 256],
                "shard_dim": 0,
                "dtype_bytes": 4,
                "placements": {0: {"start": 0, "end": 256}, 1: {"start": 256, "end": 512}},
            },
        },
        2,
        1,
    ),
    (
        {
            "w1": {
                "shape": [64, 64],
                "shard_dim": 1,
                "dtype_bytes": 4,
                "placements": {0: {"start": 0, "end": 32}, 1: {"start": 32, "end": 64}},
            },
        },
        2,
        0,
    ),
]

TREES_AND_WRAP_SEQUENCES = [
    (
        {
            "should_shard": True,
            "children": {
                "enc": {
                    "should_shard": True,
                    "children": {
                        "l0": {"should_shard": True, "children": {}},
                        "l1": {"should_shard": True, "children": {}},
                    },
                },
            },
        },
        ["enc.l0", "enc.l1", "enc", ""],
    ),
    (
        {
            "should_shard": True,
            "children": {
                "enc": {
                    "should_shard": True,
                    "children": {
                        "l0": {"should_shard": True, "children": {}},
                        "l1": {"should_shard": True, "children": {}},
                    },
                },
            },
        },
        ["", "enc", "enc.l0", "enc.l1"],
    ),
    (
        {
            "should_shard": True,
            "children": {
                "a": {
                    "should_shard": True,
                    "children": {"sub_a": {"should_shard": True, "children": {}}},
                },
                "b": {"should_shard": True, "children": {}},
            },
        },
        ["a.sub_a", "a", "b", ""],
    ),
    (
        {
            "should_shard": True,
            "children": {
                "a": {
                    "should_shard": True,
                    "children": {"sub_a": {"should_shard": True, "children": {}}},
                },
                "b": {"should_shard": True, "children": {}},
            },
        },
        ["a", "a.sub_a", "b", ""],
    ),
    (
        {
            "should_shard": True,
            "children": {
                "block": {
                    "should_shard": True,
                    "children": {
                        "attn": {"should_shard": True, "children": {}},
                        "mlp": {"should_shard": True, "children": {}},
                    },
                },
            },
        },
        ["block.attn", "block", "block.mlp", ""],
    ),
]


def ref_verify_state_dict(param_specs, world_size, rank):
    from fsdp_verify.state_dict import verify_communication_free_state_dict
    return verify_communication_free_state_dict(param_specs, world_size, rank)


def ref_analyze_wrap(model_tree, wrap_sequence):
    from fsdp_verify.wrap_order import analyze_wrap_violations
    return analyze_wrap_violations(model_tree, wrap_sequence)


def ref_reconstruct(model_tree):
    from fsdp_verify.reconstruct import reconstruct_fully_shard_sequence
    return reconstruct_fully_shard_sequence(model_tree)
