import os
import tempfile
import ref


def check(workdir):
    out = {"valid_manifests_passed": 0.0, "corrupted_manifests_rejected": 0.0}
    from edge_cache.manifest import verify_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = ref.generate_test_environment(tmpdir, num_files=3)
        if verify_manifest(manifest_path, tmpdir) is True:
            out["valid_manifests_passed"] = 1.0
        else:
            out["_note"] = "Valid manifest was rejected by verify_manifest"
            return out

        target_file = os.path.join(tmpdir, "artifact_0.bin")
        with open(target_file, "ab") as f:
            f.write(b"corruption_data")

        if verify_manifest(manifest_path, tmpdir) is False:
            out["corrupted_manifests_rejected"] = 1.0
        else:
            out["_note"] = "Corrupted artifact was not detected by verify_manifest"

    return out
