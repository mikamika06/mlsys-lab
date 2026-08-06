import ref
import random

def check(workdir):
    try:
        from kvblocks.mapping import gather_slot_mapping
        from kvblocks.trace import find_leaked_blocks
    except ImportError:
        return {"slots_match": 0.0, "leaks_match": 0.0}

    seq_lens = [5, 10, 2]
    block_tables = [[10, 11], [20, 21, 22], [30]]
    block_size = 4

    want_slots = ref.gather_slot_mapping(seq_lens, block_tables, block_size)
    try:
        got_slots = gather_slot_mapping(seq_lens, block_tables, block_size)
    except NotImplementedError:
        got_slots = None

    trace = ref.gen_trace()
    want_leaks = ref.find_leaked_blocks(trace)
    try:
        got_leaks = find_leaked_blocks(trace)
    except NotImplementedError:
        got_leaks = None

    return {
        "slots_match": 1.0 if got_slots == want_slots else 0.0,
        "leaks_match": 1.0 if got_leaks == want_leaks else 0.0
    }
