class EngineCache:
    def __init__(self):
        raise NotImplementedError

    def compute_hash(self, subgraph):
        raise NotImplementedError

    def has(self, subgraph_hash):
        raise NotImplementedError

    def get(self, subgraph_hash):
        raise NotImplementedError

    def put(self, subgraph_hash, engine_data):
        raise NotImplementedError

    def build_or_load(self, subgraph, builder_fn):
        raise NotImplementedError
