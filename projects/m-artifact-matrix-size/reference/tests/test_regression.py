"""Learner regression tests."""

from artifact_matrix.container_patch import resolve_container_patch


def test_container_patch_drift():
    res_strict = resolve_container_patch("10.2.1", "10.2.4", "strict")
    assert not res_strict["compatible"]
    assert res_strict["action"] == "reject_patch_mismatch"

    res_alias = resolve_container_patch("10.2.1", "10.2.4", "auto_patch_alias")
    assert res_alias["compatible"]
    assert res_alias["resolved_version"] == "10.2.1"
    assert res_alias["action"] == "aliased_to_container_patch"

    res_major_diff = resolve_container_patch("10.2.1", "10.3.1", "auto_patch_alias")
    assert not res_major_diff["compatible"]
    assert res_major_diff["action"] == "reject_major_minor_mismatch"
