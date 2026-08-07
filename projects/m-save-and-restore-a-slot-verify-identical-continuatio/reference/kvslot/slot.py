def save_slot(tokens, kv_state):
    return {"tokens": list(tokens), "kv": {k: list(v) for k, v in kv_state.items()}}

def restore_slot(slot_data):
    return slot_data

def verify_continuation(orig_tokens, restored_tokens):
    return list(orig_tokens) == list(restored_tokens)
