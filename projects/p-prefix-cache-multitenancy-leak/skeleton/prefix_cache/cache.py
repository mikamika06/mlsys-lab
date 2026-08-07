class PrefixCache:
    def __init__(self, capacity=100, isolate=True):
        raise NotImplementedError

    def insert(self, tokens, tenant_id="default", system_prefixes=None):
        raise NotImplementedError

    def lookup(self, tokens, tenant_id="default", system_prefixes=None):
        raise NotImplementedError
