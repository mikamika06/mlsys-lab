OUTCOME_VALID = "VALID_EAGLE_CONFIG"
OUTCOME_DRAFT_MODEL_MISMATCH = "ERR_DRAFT_MODEL_MISMATCH"
OUTCOME_HEAD_DIM_MISMATCH = "ERR_HEAD_DIM_MISMATCH"
OUTCOME_MAX_DRAFT_EXCEEDED = "ERR_MAX_DRAFT_EXCEEDED"
OUTCOME_SPEC_METHOD_UNSUPPORTED = "ERR_SPEC_METHOD_UNSUPPORTED"
OUTCOME_VOCAB_SIZE_MISMATCH = "ERR_VOCAB_SIZE_MISMATCH"


def diagnose_speculative_config(cfg):
    method = cfg.get("speculative_method")
    if method != "eagle3":
        return OUTCOME_SPEC_METHOD_UNSUPPORTED

    num_spec = cfg.get("num_speculative_tokens", 0)
    max_spec = cfg.get("max_model_len_speculative", 128)
    if num_spec <= 0 or num_spec > max_spec:
        return OUTCOME_MAX_DRAFT_EXCEEDED

    target_arch = cfg.get("target_model_arch")
    draft_arch = cfg.get("draft_model_arch")
    if not draft_arch or not draft_arch.startswith(target_arch):
        return OUTCOME_DRAFT_MODEL_MISMATCH

    if cfg.get("target_head_dim") != cfg.get("draft_head_dim"):
        return OUTCOME_HEAD_DIM_MISMATCH

    if cfg.get("target_vocab_size") != cfg.get("draft_vocab_size"):
        return OUTCOME_VOCAB_SIZE_MISMATCH

    return OUTCOME_VALID
