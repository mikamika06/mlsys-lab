import copy

def save_slot(slot):
    return copy.deepcopy(slot)

def restore_slot(slot, state):
    slot.clear()
    slot.update(copy.deepcopy(state))
    return slot

def verify_continuation(slot, continuation_fn):
    return continuation_fn(slot)
