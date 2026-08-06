class BlockMaskCache:
    def __init__(self):
        raise NotImplementedError

    def get_or_create(self, seq_len, block_size, factory_fn):
        raise NotImplementedError
