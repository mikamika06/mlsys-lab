class ManifestError(Exception):
    """Raised when artifact validation fails."""
    pass


def verify_manifest(manifest: dict, artifact_bytes: bytes, target_arch: str) -> bool:
    """Validate artifact hash and target architecture against manifest specs."""
    raise NotImplementedError
