def gather_slot_mapping(seq_lens: list[int], block_tables: list[list[int]], block_size: int) -> list[int]:
    slots = []
    for l, table in zip(seq_lens, block_tables):
        for i in range(l):
            slots.append(table[i // block_size] * block_size + (i % block_size))
    return slots
