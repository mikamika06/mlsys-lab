def save_slot(tokens, kv_state):
    raise NotImplementedError

def restore_slot(slot_data):
    raise NotImplementedError

def verify_continuation(orig_tokens, restored_tokens):
    raise NotImplementedError
