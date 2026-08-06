import os
import tempfile
from coreflow.compatibility import resolve_availability


def test_compatibility_resolution():
    spec = {
        "minimum_deployment_target": "iOS17",
        "operators": [{"name": "CustomOp", "min_version": "iOS17"}]
    }
    res = resolve_availability(spec, "iOS16")
    assert res["success"] is True


def test_package_inspection_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "Data"))
        os.makedirs(os.path.join(tmpdir, "Weights"))
        with open(os.path.join(tmpdir, "Manifest.json"), "w") as f:
            f.write("{}")
        from coreflow.inspector import inspect_package
        res = inspect_package(tmpdir)
        assert res["structure_valid"] is True
