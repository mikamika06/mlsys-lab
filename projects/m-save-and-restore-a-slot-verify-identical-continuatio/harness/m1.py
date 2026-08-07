import ref

def check(workdir):
    from kvslot.slot import save_slot, restore_slot, verify_continuation
    tokens = [1, 2, 3, 4, 5]
    kv = {"k": [0.1, 0.2], "v": [0.3, 0.4]}
    saved = save_slot(tokens, kv)
    restored = restore_slot(saved)
    valid = verify_continuation(tokens, restored["tokens"])
    out = {"slot_matched": 1.0 if valid else 0.0}
    return out
