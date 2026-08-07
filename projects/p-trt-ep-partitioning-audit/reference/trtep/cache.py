import hashlib
import json


class EngineCache:
    def __init__(self):
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def compute_hash(self, subgraph):
        nodes_data = []
        for node in subgraph.nodes:
            nodes_data.append({
                "op": node.op_type,
                "in": sorted(node.inputs),
                "out": sorted(node.outputs)
            })
        serialized = json.dumps(nodes_data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def has(self, subgraph_hash):
        return subgraph_hash in self._cache

    def get(self, subgraph_hash):
        if subgraph_hash in self._cache:
            self.hits += 1
            return self._cache[subgraph_hash]
        self.misses += 1
        return None

    def put(self, subgraph_hash, engine_data):
        self._cache[subgraph_hash] = engine_data

    def build_or_load(self, subgraph, builder_fn):
        h = self.compute_hash(subgraph)
        cached = self.get(h)
        if cached is not None:
            return cached, True
        engine = builder_fn(subgraph)
        self.put(h, engine)
        return engine, False
