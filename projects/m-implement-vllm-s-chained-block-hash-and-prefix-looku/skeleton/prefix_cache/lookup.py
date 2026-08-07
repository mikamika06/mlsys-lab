class PrefixCacheManager:

    def __init__(self, block_size):
        raise NotImplementedError

    def insert_sequence(self, token_ids):
        raise NotImplementedError

    def lookup_prefix(self, token_ids):
        raise NotImplementedError


def compute_trace_hit_rate(requests, block_size):
    raise NotImplementedError
