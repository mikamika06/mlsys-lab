def compute_dynamic_ntk_base(base, seq_len, max_position_embeddings):
    raise NotImplementedError


def compute_yarn_parameters(base, seq_len, max_position_embeddings, original_max_position_embeddings, beta_fast, beta_slow, mscale):
    raise NotImplementedError


def compute_llama3_scaling(base, seq_len, max_position_embeddings, original_max_position_embeddings, factor, low_freq_factor, high_freq_factor):
    raise NotImplementedError
