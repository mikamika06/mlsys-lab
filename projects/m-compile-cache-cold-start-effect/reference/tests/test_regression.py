import os
import tempfile
import json
import hashlib
from edge_cache.manifest import verify_manifest


def test_manifest_verification_on_corrupted_artifact():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "model.bin")
        data = b"correct_model_bytes"
        with open(artifact_path, "wb") as f:
            f.write(data)

        expected_hash = hashlib.sha256(data).hexdigest()
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"files": {"model.bin": expected_hash}}, f)

        assert verify_manifest(manifest_path, tmpdir) is True

        with open(artifact_path, "wb") as f:
            f.write(b"corrupted_model_bytes")

        assert verify_manifest(manifest_path, tmpdir) is False
