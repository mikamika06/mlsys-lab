class IdentityCache:
    """Identity-keyed cache; an entry auto-evicts when its key object dies."""

    def __init__(self):
        raise NotImplementedError('your code here')

    def put(self, key, value):
        raise NotImplementedError('your code here')

    def get(self, key, default=None):
        raise NotImplementedError('your code here')

    def __len__(self):
        raise NotImplementedError('your code here')
