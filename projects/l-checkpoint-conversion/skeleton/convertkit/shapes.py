QUANTISED = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K",
             "Q5_K", "Q6_K", "Q8_K", "IQ4_NL", "IQ4_XS", "MXFP4"}


def target_shape(ggml_shape):
    raise NotImplementedError


def is_quantised(ggml_type):
    raise NotImplementedError


def attention_shapes(meta, prefix):
    """{hidden, heads, kv_heads, head_dim, q_out, kv_out, grouped, group_size}.

    The head dimension is a metadata field, not hidden // heads. Both fixtures
    disagree with the division, and a converter that divides builds projections
    of the wrong size.
    """
    raise NotImplementedError


def check_attention(tensors, meta, prefix, layer=0):
    """{expected, problems}: do the stored q, k and v shapes agree with what
    the metadata implies."""
    raise NotImplementedError
