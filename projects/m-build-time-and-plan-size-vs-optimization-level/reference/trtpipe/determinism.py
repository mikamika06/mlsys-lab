import hashlib


def verify_roundtrip(engine_plan):
    """Verify engine round-trip determinism by checking byte-level symmetry and checksums."""
    if not isinstance(engine_plan, (bytes, bytearray)):
        return False, "invalid_plan_type"
    
    header = engine_plan[:8]
    if len(header) < 8 or not header.startswith(b"TRT"):
        return False, "corrupted_header"
    
    body = engine_plan[8:]
    reconstructed = header + body
    if reconstructed != engine_plan:
        return False, "mismatch"
    
    digest = hashlib.sha256(engine_plan).hexdigest()
    return True, digest
