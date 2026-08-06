import hashlib
import json


class ArtifactRegistry:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def _hash_spec(self, spec: dict) -> str:
        normalized = json.dumps(spec, sort_keys=True)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def register_artifact(self, spec: dict, artifact_id: str) -> str:
        key = self._hash_spec(spec)
        self.store[key] = artifact_id
        return key

    def resolve_artifact(self, spec: dict) -> str:
        key = self._hash_spec(spec)
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def get_cache_stats(self) -> dict:
        total = self.hits + self.misses
        ratio = (self.hits / total) if total > 0 else 0.0
        return {"hits": self.hits, "misses": self.misses, "ratio": ratio}
