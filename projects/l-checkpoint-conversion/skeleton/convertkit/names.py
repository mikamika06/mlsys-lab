def to_target(name, experts=0):
    """The target name for one GGUF tensor.

    A string when it is one to one, a list when a fused expert tensor fans out
    to one target per expert, None when there is no counterpart. Returning a
    single name for a fused expert tensor is worse than returning None: it
    silently keeps one expert and drops the rest.

    The convention is the one in the MLX fixture: `layers.0.self_attn.q_proj.
    weight`, `layers.0.mlp.gate_proj.weight`, `layers.0.input_layernorm.
    weight`, `embed_tokens.weight`, `norm.weight`, `lm_head.weight`. The full
    table is in the brief.
    """
    raise NotImplementedError


def map_index(tensors, experts=0):
    """{mapped, fanned_out, unmapped, target_count}."""
    raise NotImplementedError


def layer_targets(mapping, layer):
    """The set of target names belonging to one layer."""
    raise NotImplementedError
