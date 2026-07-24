def measure_blocks(seq_len: int, block_size: int):
    """
    Return the number of blocks needed to store seq_len tokens in blocks of size block_size,
    a mapping array where each element is the global physical slot index for that logical token,
    and the total unused slots (internal fragmentation).
    """
    import numpy as np

    # Ceiling division to get the required number of blocks
    num_blocks = -(-seq_len // block_size)  # equivalent to math.ceil(seq_len / block_size)

    # Total waste in the last block
    waste = num_blocks * block_size - seq_len

    # Global slot indices for each logical token
    slot_mapping = np.arange(num_blocks * block_size, dtype=np.int64)[:seq_len]

    return num_blocks, slot_mapping, waste
