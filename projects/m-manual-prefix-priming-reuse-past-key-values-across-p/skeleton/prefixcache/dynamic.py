class DynamicCache:
    """Dynamic key-value cache supporting layer updates, cropping, and slicing."""

    def __init__(self, key_cache=None, value_cache=None):
        raise NotImplementedError

    def update(self, key_states, value_states, layer_idx):
        raise NotImplementedError

    def get_seq_length(self, layer_idx=0):
        raise NotImplementedError

    def crop(self, max_length):
        raise NotImplementedError

    def slice(self, start, end):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError
