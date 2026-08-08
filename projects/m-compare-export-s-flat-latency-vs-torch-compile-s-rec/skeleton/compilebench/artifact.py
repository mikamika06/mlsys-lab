class ArtifactError(Exception):
    """Base exception for export artifact issues."""


class InvalidMagicError(ArtifactError):
    """Raised when artifact magic bytes do not match expected header."""


class TruncatedArtifactError(ArtifactError):
    """Raised when artifact data ends prematurely."""


class CorruptedArtifactError(ArtifactError):
    """Raised when artifact checksum or offset integrity check fails."""


def serialize_export_artifact(graph_spec):
    """Serialize graph specification into binary export format."""
    raise NotImplementedError


def deserialize_export_artifact(data):
    """Validate and deserialize binary export artifact."""
    raise NotImplementedError
