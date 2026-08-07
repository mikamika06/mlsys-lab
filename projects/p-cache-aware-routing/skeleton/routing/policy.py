class CacheAwarePolicy:
    def __init__(self, num_replicas, load_weight=0.5):
        raise NotImplementedError

    def route(self, prompt, replica_states):
        raise NotImplementedError
