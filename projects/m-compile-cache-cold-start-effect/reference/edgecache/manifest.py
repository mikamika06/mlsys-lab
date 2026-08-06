import hashlib


class ManifestError(Exception):
    """Raised when artifact validation fails."""
    pass


def verify_manifest(manifest: dict, artifact_bytes: bytes, target_arch: str) -> bool:
    """Validate artifact hash and target architecture against manifest specs."""
    if manifest.get("target_arch") != target_arch:
        raise ManifestError("Architecture mismatch")

    computed_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if manifest.get("sha256") != computed_hash:
        raise ManifestError("SHA256 checksum mismatch")

    return True
