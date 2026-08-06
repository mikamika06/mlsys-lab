"""Validation of draft and target model compatibility."""


def is_valid_draft_target_pair(draft_meta: dict, target_meta: dict) -> bool:
    """Determine if a draft model and target model form a valid speculative decoding pair."""
    if draft_meta.get("vocab_size") != target_meta.get("vocab_size"):
        return False
    if draft_meta.get("bos_token_id") != target_meta.get("bos_token_id"):
        return False
    if draft_meta.get("eos_token_id") != target_meta.get("eos_token_id"):
        return False

    draft_tokens = draft_meta.get("tokens")
    target_tokens = target_meta.get("tokens")
    if draft_tokens is not None and target_tokens is not None:
        if draft_tokens != target_tokens:
            return False

    draft_add_eos = draft_meta.get("add_eos_token", True)
    target_add_eos = target_meta.get("add_eos_token", True)
    if draft_add_eos != target_add_eos:
        return False

    return True
