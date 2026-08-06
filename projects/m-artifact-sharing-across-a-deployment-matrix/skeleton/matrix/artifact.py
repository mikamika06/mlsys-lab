class ArtifactRegistry:
    def __init__(self):
        pass

    def register_artifact(self, spec: dict, artifact_id: str) -> str:
        raise NotImplementedError

    def resolve_artifact(self, spec: dict) -> str:
        raise NotImplementedError

    def get_cache_stats((self) -> dict:
        raise NotImplementedError
